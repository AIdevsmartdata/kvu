/* SPDX-License-Identifier: Apache-2.0
 *
 * FAUX runtime HIP + hiprtc pour les essais de kvu (tests/gpu_faute_rc.rs) : AUCUN calcul, aucun GPU. Les tampons « device » sont de la
 * mémoire de l'hôte, les noyaux ne font rien, et les pannes sont pilotées par l'environnement. Compilé par l'essai lui-même, puis
 * trouvé par kvu (dlopen « libamdhip64.so » et « libhiprtc.so ») grâce à LD_LIBRARY_PATH : le chemin d'erreur essayé est donc celui du
 * vrai backend HIP de kvu (heritage::wrap, gpu::hip, gpu::mod, su2, cli), du code retour d'une fonction du runtime jusqu'au code de
 * sortie du processus.
 *
 * Un faux runtime ne calcule rien : toute commande qui va au bout avec lui rend des zéros, jamais un résultat. Il ne sert qu'aux essais
 * de code de sortie et de message d'erreur.
 *
 * Détecteur de durée de vie (activé par défaut) : le faux runtime compte les opérations ASYNCHRONES encore « en vol » (lancements de noyaux,
 * copies device -> device, mises à zéro) depuis le dernier appel qui attend l'appareil (hipStreamSynchronize, hipDeviceSynchronize). Un
 * hipFree ou un hipModuleUnload qui arrive avec des opérations en vol est une VIOLATION : sur un vrai GPU c'est une faute de page
 * (tampon libéré sous un noyau en cours : client TCP ; code objet déchargé sous un noyau en cours : client SQC) que le processus peut ne
 * jamais voir. Le faux runtime ne s'appuie volontairement PAS sur la synchronisation implicite de hipFree : le code de kvu doit
 * synchroniser lui-même, et vérifier le code de retour, AVANT toute libération. Les violations sont comptées (`viol=` du fichier de comptes) et
 * écrites sur stderr ; FAUX_HIP_STRICT=0 désactive le détecteur.
 *
 * Variables d'environnement :
 *   FAUX_HIP_PANNE   <fonction>:<rang>:<code>[:collant]
 *       fonction : sync | devsync | dtoh | htod | launch   le <rang>-ième appel (à partir de 1) de hipStreamSynchronize, hipDeviceSynchronize,
 *                                                  hipMemcpyDtoHAsync, hipMemcpyHtoDAsync, hipModuleLaunchKernel rend <code> (700 : accès
 *                                                  mémoire illégal, 719 : échec de lancement) ;
 *                  apres_launch                    le noyau « fait sa faute » en s'exécutant : le PREMIER appel de complétion (sync, dtoh
 *                                                  ou htod) qui suit le <rang>-ième lancement rend <code>, comme un vrai runtime ;
 *       collant : une fois déclenchée, la panne se répète à chaque appel de complétion ou de lancement suivant.
 *       À la panne, la ligne « Queue error - HSA_STATUS_ERROR_MEMORY_FAULT » est écrite sur stderr, comme le fait le runtime réel.
 *   FAUX_HIP_COMPTES chemin d'un fichier où la bibliothèque ajoute, en sortant, « sync=N devsync=N dtoh=N htod=N launch=N free=N unload=N viol=N ».
 *   FAUX_HIP_STRICT  0 : pas de détecteur de durée de vie (1 par défaut).
 */
#define _GNU_SOURCE
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int hipError_t;
typedef void *hipStream_t;
typedef void *hipModule_t;
typedef void *hipFunction_t;
typedef void *hiprtcProgram;

enum { F_SYNC, F_DEVSYNC, F_DTOH, F_HTOD, F_LAUNCH, F_APRES_LAUNCH, F_NB };
static const char *NOMS[F_NB] = {"sync", "devsync", "dtoh", "htod", "launch", "apres_launch"};

static unsigned long appels[F_APRES_LAUNCH];   /* sync, devsync, dtoh, htod, launch */
static unsigned long en_vol;          /* opérations asynchrones lancées depuis le dernier appel qui attend l'appareil */
static unsigned long violations;      /* hipFree / hipModuleUnload avec des opérations en vol */
static unsigned long liberations, dechargements;
static int strict = 1;
static int panne_fonction = -1;
static unsigned long panne_rang;
static int panne_code;
static int panne_collante;
static int panne_declenchee;          /* la panne a eu lieu (état collant si panne_collante) */
static int panne_en_attente;          /* apres_launch : le noyau fautif est parti, la faute n'est pas encore rapportée */
static int initialise;

static void init(void)
{
    const char *s;
    char nom[32], collant[16] = "";
    unsigned long rang;
    int code, n, i;

    if (initialise)
        return;
    initialise = 1;
    s = getenv("FAUX_HIP_STRICT");
    if (s && strcmp(s, "0") == 0)
        strict = 0;
    s = getenv("FAUX_HIP_PANNE");
    if (!s || !*s)
        return;
    n = sscanf(s, "%31[a-z_]:%lu:%d:%15s", nom, &rang, &code, collant);
    if (n < 3) {
        fprintf(stderr, "faux_hip : FAUX_HIP_PANNE invalide : %s\n", s);
        exit(97);
    }
    for (i = 0; i < F_NB; i++)
        if (strcmp(nom, NOMS[i]) == 0)
            panne_fonction = i;
    if (panne_fonction < 0) {
        fprintf(stderr, "faux_hip : fonction inconnue dans FAUX_HIP_PANNE : %s\n", s);
        exit(97);
    }
    panne_rang = rang;
    panne_code = code;
    panne_collante = n >= 4 && strcmp(collant, "collant") == 0;
}

static hipError_t declencher(void)
{
    panne_declenchee = 1;
    panne_en_attente = 0;
    fprintf(stderr, "Queue error - HSA_STATUS_ERROR_MEMORY_FAULT\n");
    return panne_code;
}

/* Un hipFree ou un hipModuleUnload avec des opérations asynchrones en vol : violation (voir l'en-tête). */
static void liberation(const char *quoi)
{
    init();
    if (strict && en_vol) {
        violations++;
        fprintf(stderr, "faux_hip : %s avec %lu opération(s) asynchrone(s) encore en vol (aucune synchronisation avant la libération)\n", quoi, en_vol);
    }
}

/* Appel de complétion (sync, devsync, dtoh, htod) ou de lancement : rend le code de la panne si c'est son tour. Un appel qui attend l'appareil
 * (sync, devsync) remet à zéro les opérations en vol ; une panne n'a pas besoin de le faire : le processus s'arrête sur l'erreur. */
static hipError_t passage(int f)
{
    init();
    appels[f]++;
    if (f == F_SYNC || f == F_DEVSYNC)
        en_vol = 0;
    if (panne_declenchee && panne_collante)
        return panne_code;
    if (panne_fonction == f && appels[f] == panne_rang)
        return declencher();
    if (f != F_LAUNCH && panne_en_attente)
        return declencher();
    if (f == F_LAUNCH) {
        en_vol++;
        if (panne_fonction == F_APRES_LAUNCH && appels[F_LAUNCH] == panne_rang)
            panne_en_attente = 1;
    }
    return 0;
}

__attribute__((destructor)) static void fin(void)
{
    const char *chemin = getenv("FAUX_HIP_COMPTES");
    FILE *f;

    if (!chemin || !*chemin || (appels[F_SYNC] + appels[F_DEVSYNC] + appels[F_DTOH] + appels[F_HTOD] + appels[F_LAUNCH]) == 0)
        return;
    f = fopen(chemin, "a");
    if (!f)
        return;
    fprintf(f, "sync=%lu devsync=%lu dtoh=%lu htod=%lu launch=%lu free=%lu unload=%lu viol=%lu\n", appels[F_SYNC], appels[F_DEVSYNC], appels[F_DTOH],
            appels[F_HTOD], appels[F_LAUNCH], liberations, dechargements, violations);
    fclose(f);
}

/* ---------------------------------------------------------------- HIP */
hipError_t hipInit(unsigned int flags) { (void)flags; return 0; }
hipError_t hipGetDeviceCount(int *n) { *n = 1; return 0; }
hipError_t hipSetDevice(int d) { (void)d; return 0; }
hipError_t hipStreamCreate(hipStream_t *s) { *s = malloc(1); return 0; }
hipError_t hipStreamDestroy(hipStream_t s) { free(s); return 0; }
hipError_t hipStreamSynchronize(hipStream_t s) { (void)s; return passage(F_SYNC); }
hipError_t hipDeviceSynchronize(void) { return passage(F_DEVSYNC); }
hipError_t hipGetLastError(void) { return 0; }
hipError_t hipPeekAtLastError(void) { return 0; }
hipError_t hipMalloc(void **p, size_t n) { *p = calloc(1, n ? n : 1); return *p ? 0 : 2; }
hipError_t hipFree(void *p)
{
    liberation("hipFree");
    liberations++;
    en_vol = 0;   /* le détecteur a jugé ; l'appareil de test, lui, n'a rien en cours */
    free(p);
    return 0;
}

hipError_t hipMemsetD8Async(void *p, unsigned char v, size_t n, hipStream_t s)
{
    (void)s;
    init();
    en_vol++;
    memset(p, v, n);
    return 0;
}

hipError_t hipMemcpyHtoDAsync(void *dst, const void *src, size_t n, hipStream_t s)
{
    hipError_t r = passage(F_HTOD);
    (void)s;
    if (r)
        return r;
    memcpy(dst, src, n);
    return 0;
}

hipError_t hipMemcpyDtoHAsync(void *dst, void *src, size_t n, hipStream_t s)
{
    hipError_t r = passage(F_DTOH);
    (void)s;
    if (r)
        return r;
    memcpy(dst, src, n);
    return 0;
}

hipError_t hipMemcpyDtoDAsync(void *dst, void *src, size_t n, hipStream_t s)
{
    (void)s;
    init();
    en_vol++;
    memcpy(dst, src, n);
    return 0;
}
hipError_t hipModuleLoadData(hipModule_t *m, const void *image) { (void)image; *m = malloc(1); return 0; }

hipError_t hipModuleUnload(hipModule_t m)
{
    liberation("hipModuleUnload");
    dechargements++;
    en_vol = 0;
    free(m);
    return 0;
}
hipError_t hipModuleGetFunction(hipFunction_t *f, hipModule_t m, const char *nom) { (void)m; (void)nom; *f = malloc(1); return 0; }

hipError_t hipModuleLaunchKernel(hipFunction_t f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz,
                                 unsigned shmem, hipStream_t s, void **params, void **extra)
{
    (void)f; (void)gx; (void)gy; (void)gz; (void)bx; (void)by; (void)bz; (void)shmem; (void)s; (void)params; (void)extra;
    return passage(F_LAUNCH);
}

hipError_t hipFuncSetAttribute(const void *f, int attr, int valeur) { (void)f; (void)attr; (void)valeur; return 0; }

const char *hipGetErrorString(hipError_t e)
{
    switch (e) {
    case 0: return "no error";
    case 700: return "an illegal memory access was encountered";
    case 719: return "unspecified launch failure";
    default: return "unknown error";
    }
}

hipError_t hipMemGetInfo(size_t *libre, size_t *total) { *libre = (size_t)64 << 30; *total = (size_t)128 << 30; return 0; }
hipError_t hipRuntimeGetVersion(int *v) { *v = 70000001; return 0; }
hipError_t hipDeviceGetName(char *nom, int longueur, int d) { (void)d; snprintf(nom, (size_t)longueur, "GPU factice (essai de kvu)"); return 0; }

/* ---------------------------------------------------------------- hiprtc */
int hiprtcCreateProgram(hiprtcProgram *p, const char *src, const char *nom, int n, const char **en_tetes, const char **noms)
{
    (void)src; (void)nom; (void)n; (void)en_tetes; (void)noms;
    *p = calloc(1, 16);
    return 0;
}
int hiprtcCompileProgram(hiprtcProgram p, int n, const char **options) { (void)p; (void)n; (void)options; return 0; }
int hiprtcGetProgramLogSize(hiprtcProgram p, size_t *n) { (void)p; *n = 1; return 0; }
int hiprtcGetProgramLog(hiprtcProgram p, char *tampon) { (void)p; tampon[0] = 0; return 0; }
int hiprtcGetCodeSize(hiprtcProgram p, size_t *n) { (void)p; *n = 64; return 0; }
int hiprtcGetCode(hiprtcProgram p, char *tampon) { (void)p; memset(tampon, 0x5a, 64); return 0; }
int hiprtcDestroyProgram(hiprtcProgram *p) { free(*p); *p = NULL; return 0; }
const char *hiprtcGetErrorString(int r) { return r ? "erreur hiprtc factice" : "HIPRTC_SUCCESS"; }
int hiprtcVersion(int *majeur, int *mineur) { *majeur = 7; *mineur = 0; return 0; }
