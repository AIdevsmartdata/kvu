// prelude_hip.h — prélude HIP (hiprtc), inséré en tête de chaque source assemblée pour le backend HIP.
// Propriétaire : T2. Il fournit __global__, __shared__, atomicAdd, sqrt/log/cos (double et float)
// par l'en-tête d'exécution HIP, et pose la macro KV_HIP pour les rares branches spécifiques.
// Les noyaux restent `extern "C"` : la recherche par nom (hipModuleGetFunction) est une chaîne simple.
#include <hip/hip_runtime.h>
#define KV_HIP 1
