# Dérivation indépendante — réponses

## Notations communes

On pose \(X=[\omega]_\times\), donc \(Xv=\omega\times v\), et
\[
R(W)=I-\sin(2\theta)[n]_\times+(1-\cos2\theta)[n]_\times^2,
\qquad W(i\varepsilon\cdot\sigma)W^{-1}=i(R(W)\varepsilon)\cdot\sigma .
\]
Les \(I_\nu\) ci-dessous sont les fonctions de Bessel modifiées de première espèce. La métrique imposée donne, sur SU(2),
\[
dV_{\rm SU(2)}=\sin^2\theta\,d\theta\,d\Omega_2,\qquad
\operatorname{Vol}{\rm SU(2)}=2\pi^2,\qquad dU=\frac{dV_{\rm SU(2)}}{2\pi^2}.
\]

## Q1 — Désintégration et dynamique contrainte

On travaille sur un ouvert où tous les \(U_0^{-1}U(\Gamma)\) évitent \(-1\), où \(\bar U\) est lisse et où \(J=D\bar U\) est de rang maximal \(3n\). On note
\[
G(U)=JJ^T,\qquad d\sigma_V=\text{mesure de Hausdorff riemannienne sur }F_V.
\]
La formule de co-aire donne
\[
dV_{{\rm SU(2)}^N}=\frac{d\sigma_V\,dV_{{\rm SU(2)}^n}}{\sqrt{\det G}}.
\]
En tenant compte des Dirac normalisés par Haar, la loi conditionnelle est donc
\[
\boxed{\;
\mu(dU\mid \bar U=V)=\frac{1}{Z(V)}\,
 e^{-S(U)}(\det G(U))^{-1/2}\,d\sigma_V(U),\qquad
Z(V)=\int_{F_V}e^{-S}(\det G)^{-1/2}d\sigma_V .
\;}
\]
La marginale de \(V\), relativement au produit des mesures de Haar et pour le poids de Gibbs non normalisé, est
\[
\boxed{\;m(V)=(2\pi^2)^{-(N-n)}Z(V).\;}
\]
Si le poids de Gibbs initial est normalisé par \(Z_{\rm tot}\), il faut encore diviser cette marginale par \(Z_{\rm tot}\).

**Trivialisation et jauge.** Remplacer, lien par lien, les perturbations gauches par des perturbations droites multiplie les coordonnées tangentes par des rotations \(R(U_\ell)^{\pm1}\), et fait de même au but. Les déterminants de ces changements de base ont valeur absolue 1, donc \(\det JJ^T\) est inchangé. Sous une transformation de jauge,
\[
U_0^{-1}U(\Gamma)\mapsto U_0^{-1}U(\Gamma),\qquad
\bar U_c\mapsto g_{x_c}\bar U_c g_{y_c}^{-1},
\]
et \(J_{c\ell}\mapsto R(g_{x_c})J_{c\ell}R(g_{x_\ell})^{-1}\). Le facteur est donc invariant de jauge.

**RATTLE avec masse \(M\).** Pour la mesure contrainte de RATTLE dans les coordonnées ambiantes, avec \(M=M^T>0\) constante et
\[
A(U)=JM^{-1}J^T,
\]
la densité configurée sur la fibre, avant correction, est proportionnelle à
\[
e^{-S}\,\frac{\sqrt{\det M}}{\sqrt{\det A\,\det G}}.
\]
Pour obtenir exactement \(\mu(\cdot\mid V)\), il faut ainsi utiliser
\[
\boxed{\;
S_{\rm eff}=S+\frac12\log\det M-\frac12\log\det(JM^{-1}J^T)
\;}
\]
(ou omettre le terme constant \(\tfrac12\log\det M\)). Pour \(M=I\), aucun potentiel additionnel n’est nécessaire : RATTLE fournit déjà le facteur \(G^{-1/2}\).

## Q2 — Dérivées fermées de log et exp

Définissons
\[
a(\theta)=\frac{\sin^2\theta}{\theta^2},\qquad
b_E(\theta)=\frac{\theta-\sin\theta\cos\theta}{\theta^3},\qquad
b_L(\theta)=\frac1{\theta^2}-\frac{\cot\theta}{\theta}.
\]
Alors
\[
\boxed{
\begin{aligned}
L_g(\omega)&=I+X+b_L(\theta)X^2,\\
L_d(\omega)&=I-X+b_L(\theta)X^2=L_g(-\omega),\\
E_g(\omega)&=I-a(\theta)X+b_E(\theta)X^2 .
\end{aligned}}
\]
En particulier \(L_g=E_g^{-1}\), et non \(L_g=E_g\). Au voisinage de zéro,
\[
L_{g,d}=I\pm X+\frac13X^2+O(\theta^2),\qquad
E_g=I-X+\frac23X^2+O(\theta^2),
\]
avec le signe supérieur pour \(L_g\). Quand \(\theta\to\pi\),
\[
E_g\to nn^T-\frac1\pi[n]_\times,
\]
tandis que la direction radiale de \(L_{g,d}\) reste égale à \(nn^T\), mais
\[
b_L(\theta)\sim\frac1{\pi(\pi-\theta)} :
\]
leurs coefficients transverses divergent. Cette divergence est celle du logarithme principal près de \(-1\).

Les différences finies centrales à 60 chiffres, avec \(h=10^{-20}\), donnent les erreurs maximales
\[
9.51\times10^{-42}\;(L_g),\qquad
5.68\times10^{-42}\;(L_d),\qquad
4.88\times10^{-42}\;(E_g).
\]

## Q3 — Jacobien de la carte de Bałaban

Pour un chemin \(\Gamma\), soit \(P_{k-1}\) le produit des maillons strictement avant l’occurrence \(k\), et \(P_k\) le produit jusqu’à cette occurrence incluse. Son jacobien de chemin \(T_\Gamma\) a les blocs
\[
\boxed{\;
(T_\Gamma)_\ell=
\sum_{k:\ell_k=\ell,\ s_k=+1}R(P_{k-1})
-\sum_{k:\ell_k=\ell,\ s_k=-1}R(P_k).
\;}
\]
Cette formule prend automatiquement en charge les répétitions et les parcours à rebours.

Pour une maille grossière \(c\), écrivons
\[
H_\Gamma=U_0^{-1}U(\Gamma)=e^{i\omega_\Gamma\cdot\sigma},\qquad
\bar\omega=\sum_\Gamma w_\Gamma\omega_\Gamma,\qquad R_0=R(U_0),
\]
et définissons
\[
B_\Gamma=R_0\,E_g(\bar\omega)\,L_g(\omega_\Gamma)\,R_0^T,\qquad
B_0=I-\sum_\Gamma w_\Gamma B_\Gamma .
\]
Alors
\[
\boxed{\;
J_{c\ell}=B_0(T_{\Gamma_0(c)})_\ell+
\sum_{\Gamma\in\mathcal G(c)}w_\Gamma B_\Gamma(T_\Gamma)_\ell .
\;}
\]
Le facteur \(E_g(\bar\omega)\) est indispensable : il convertit la variation de \(\bar\omega\) en perturbation gauche de l’exponentielle finale.

Le test utilise deux liens, trois chemins non triviaux en plus du chemin de référence, des liens répétés et à rebours. Les écarts maximaux des deux blocs avec les différences finies sont
\[
3.04\times10^{-41},\qquad1.97\times10^{-41}.
\]

## Q4 — Bloc jouet SU(2) à deux liens

### (a) Jacobien et déterminant

Posons
\[
\mathcal B=R(U_1)E_g(t\omega)L_g(\omega)R(U_1)^T.
\]
La question Q3 donne
\[
\boxed{J=(\,I-t\mathcal B\;\;t\mathcal B\,).}
\]
Dans une base où \(\omega\) est sur l’axe 3, les valeurs singulières se séparent en une direction radiale et deux directions transverses. Posons
\[
a_t=1-2t+2t^2,
\]
\[
c_t=\frac{\sin(t\theta)\cos((1-t)\theta)}{t\sin\theta},\qquad
q_t^2=\frac{\sin^2(t\theta)}{t^2\sin^2\theta},
\]
\[
\boxed{
D_t(\theta)=1-2tc_t+2t^2q_t^2
=1-\frac{2\sin(t\theta)\cos((1-t)\theta)}{\sin\theta}
+\frac{2\sin^2(t\theta)}{\sin^2\theta}.}
\]
Alors
\[
\boxed{\det JJ^T=a_tD_t(\theta)^2,\qquad
\sqrt{\det JJ^T}=\sqrt{a_t}\,D_t(\theta).}
\]
Pour \(t=\tfrac12\), \(a_t=\tfrac12\), \(D_{1/2}=(1+\cos\theta)^{-1}\), donc
\[
\boxed{\det JJ^T=\frac1{2(1+\cos\theta)^2}.}
\]

### (b) Fibre et volume induit

À \(V\) fixée, toute la fibre est paramétrée par \(W=e^{i\omega\cdot\sigma}\) :
\[
\boxed{U_1=VW^{-t}=Ve^{-it\omega\cdot\sigma},\qquad
U_2=VW^{-t}W=VW^{1-t}.}
\]
Le calcul de la métrique tirée en arrière donne
\[
\boxed{\;
d\sigma_V=\sqrt{a_t}\,\sin^2\theta\,D_t(\theta)\,d\theta\,d\Omega_2 .
\;}
\]
En particulier,
\[
\boxed{t=\tfrac12:\qquad d\sigma_V=\frac1{\sqrt2}(1-\cos\theta)\,d\theta\,d\Omega_2.}
\]

### (c) Loi sous Haar⊗Haar

Comme \(dW=(2\pi^2)^{-1}\sin^2\theta\,d\theta\,d\Omega_2\) et
\(\det E_g(\omega)=\sin^2\theta/\theta^2\), on obtient
\[
\frac{d\sigma_V}{\sqrt{\det JJ^T}}=\sin^2\theta\,d\theta\,d\Omega_2
=2\pi^2\,dW.
\]
Le facteur \((2\pi^2)^{-1}\) de la co-aire pour \(N=2,n=1\) compense exactement ce \(2\pi^2\). Par conséquent
\[
\boxed{W\mid(\bar U=V)\sim{\rm Haar}(SU(2)),}
\]
indépendamment de \(t\) et de \(V\). En argument direct indépendant, le jacobien de
\((V,W)\mapsto(VW^{-t},VW^{1-t})\), calculé en coordonnées gauches, vaut également 1 ; le changement de variables préserve donc Haar⊗Haar.

### (d) Avec \(S=-\beta\cos\theta\)

La densité radiale conditionnelle est proportionnelle à \(e^{\beta\cos\theta}\sin^2\theta\). Puisque
\(\int_0^\pi e^{\beta\cos\theta}\sin^2\theta\,d\theta=\pi I_1(\beta)/\beta\),
\[
\boxed{\left\langle\tfrac12\operatorname{Tr}W\mid V\right\rangle
=\langle\cos\theta\rangle=\frac{I_2(\beta)}{I_1(\beta)}.}
\]
Cela ne dépend ni de \(V\), ni de \(t\). Les valeurs, identiques pour \(t=\tfrac12\) et \(t=0{,}3\), sont
\[
\begin{array}{c|c}
\beta&\langle\cos\theta\rangle\\ \hline
2&0.433127426722\\
8&0.819249410356
\end{array}
\]
Valeurs non arrondies utilisées : `0.4331274267223117583171835` et `0.8192494103556205425833`.

### (e) Si l’on oublie le facteur de co-aire

Pour \(t=\tfrac12\), la mesure de Hausdorff seule donne le poids radial
\(e^{\beta\cos\theta}(1-\cos\theta)\). Ainsi
\[
\boxed{
\langle\cos\theta\rangle_{\rm H}
=\frac{\int_0^\pi\cos\theta\,e^{\beta\cos\theta}(1-\cos\theta)d\theta}
{\int_0^\pi e^{\beta\cos\theta}(1-\cos\theta)d\theta}
=\frac{2I_1-I_0-I_2}{2(I_0-I_1)}.}
\]
Les valeurs sont
\[
\beta=2:\ 0.154394686533,\qquad
\beta=8:\ 0.805069521287.
\]
Pour \(\langle1-\cos\theta\rangle\), l’écart relatif défini par
\(({\rm oubli}-{\rm correct})/{\rm correct}\) vaut
\[
\begin{array}{c|c|c|c}
\beta&\text{correct}&\text{oubli}&\text{écart relatif}\\ \hline
2&0.566872573278&0.845605313467&0.491702638880\\
8&0.180750589644&0.194930478713&0.078450029385\\
1000&0.001499624625&0.001500375564&0.000500751441
\end{array}
\]

## Q5 — Cas U(1)

### (a) Fibre compacte et branches

Sur chaque cellule où les branches des logarithmes sont fixées, la carte est une application linéaire entière pondérée :
\[
\bar\phi=Q\theta\pmod{2\pi},\qquad
Q=(1-\sum w_\Gamma)q_0+\sum w_\Gamma q_\Gamma=\sum w_\Gamma q_\Gamma,
\]
où \(q_\Gamma\) est le vecteur des signatures des liens du chemin. Donc
\[
\boxed{J=Q,\qquad \det JJ^T=\det QQ^T=\text{constante sur la cellule}.}
\]
Le facteur de Q1 disparaît de la densité conditionnelle normalisée : sur chaque composante, c’est simplement \(e^{-S}\) fois la mesure de Hausdorff.

Pour décrire exactement les composantes, choisissons des représentants angulaires et notons \(m_{c\Gamma}\in\mathbb Z\) les entiers de branche :
\[
[\phi_\Gamma-\phi_0]_{(-\pi,\pi]}=\phi_\Gamma-\phi_0+2\pi m_{c\Gamma}.
\]
Sur la cellule \(D_m\), la contrainte s’écrit
\[
Q\theta=\bar\phi+2\pi(r_c-B_c),\qquad r_c\in\mathbb Z,\qquad
B_c=\sum_\Gamma w_\Gamma m_{c\Gamma}.
\]
La fibre est donc l’union, sur les cellules \(D_m\) et les entiers \(r\), de tores linéaires de dimension \(N-n\). Si \(Z=\sum_{m,r}\int e^{-S}d\sigma_{m,r}/\sqrt{\det QQ^T}\), alors
\[
\boxed{\mu(d\theta\mid\bar\phi)=Z^{-1}
\sum_{m,r}\mathbf1_{D_m}(\theta)e^{-S(\theta)}
\frac{d\sigma_{m,r}(\theta)}{\sqrt{\det QQ^T}}.}
\]
Les poids des différentes composantes sont leurs intégrales respectives : il ne faut pas en garder une seule ni oublier les changements de branche.

### (b) Gaussien non compact avec contrainte exacte

Après fixation de jauge, supposons \(K>0\) sur l’espace restant et \(H=QK^{-1}Q^T>0\). Alors
\[
\boxed{
\theta_*=K^{-1}Q^TH^{-1}\bar\phi,\qquad
\Sigma_0=\beta^{-1}\left(K^{-1}-K^{-1}Q^TH^{-1}QK^{-1}\right).}
\]
La loi est gaussienne dégénérée sur \(Q\theta=\bar\phi\), de moyenne \(\theta_*\) et de covariance \(\Sigma_0\).

Avant fixation de jauge, si \(KZ=0\), les modes \(Z\) sont infinis et non normalisables dans le modèle non compact. L’invariance de la contrainte impose \(QZ=0\). Il faut quotienter ou fixer ces modes ; les formules ci-dessus s’appliquent alors à \(K\) restreint au complément. La moyenne dépend du représentant de jauge, mais la loi de toute observable invariante de jauge est unique.

### (c) Pénalité quadratique au lieu de la contrainte

Toujours après fixation de jauge,
\[
p(\theta\mid\bar\phi)\propto
\exp[-\tfrac\beta2\theta^TK\theta-\tfrac1{2\varepsilon^2}|Q\theta-\bar\phi|^2]
\]
est la gaussienne \(\mathcal N(m_\varepsilon,\Sigma_\varepsilon)\) avec
\[
\boxed{
\Sigma_\varepsilon=(\beta K+\varepsilon^{-2}Q^TQ)^{-1}
=\beta^{-1}\left[K^{-1}-K^{-1}Q^T(H+\beta\varepsilon^2I)^{-1}QK^{-1}\right],}
\]
\[
\boxed{
m_\varepsilon=\Sigma_\varepsilon Q^T\bar\phi/\varepsilon^2
=K^{-1}Q^T(H+\beta\varepsilon^2I)^{-1}\bar\phi.}
\]
En particulier,
\[
Qm_\varepsilon=H(H+\beta\varepsilon^2I)^{-1}\bar\phi,\qquad
Qm_\varepsilon-\bar\phi=-\beta\varepsilon^2(H+\beta\varepsilon^2I)^{-1}\bar\phi,
\]
\[
\operatorname{Cov}(Q\theta)=\varepsilon^2H(H+\beta\varepsilon^2I)^{-1}.
\]
Le biais est nul pour \(\bar\phi=0\), tend vers zéro comme \(\varepsilon^2\) lorsque \(\varepsilon\to0\), et croît avec le produit \(\beta\varepsilon^2\). On retrouve Q5(b) quand \(\varepsilon\to0\).

### (d) Bloc U(1) à deux liens

Localement, avec \(\delta=\theta_2-\theta_1\),
\[
\theta_1=\bar\phi-t\delta,\qquad \theta_2=\bar\phi+(1-t)\delta,
\qquad J=(1-t,t),\qquad JJ^T=a_t.
\]
La longueur induite est \(\sqrt{a_t}\,d\delta\), qui annule exactement \((JJ^T)^{-1/2}\). Toutes les branches ayant le même poids en \(\delta\),
\[
p(\delta\mid\bar\phi)\propto e^{\beta\cos\delta},
\]
et la réponse ne dépend pas de \(\bar\phi\) :
\[
\boxed{\langle\cos\delta\mid\bar\phi\rangle=\frac{I_1(\beta)}{I_0(\beta)}.}
\]
Numériquement,
\[
\beta=2:\ 0.697774657964,\qquad
\beta=8:\ 0.935235493529.
\]

## Q6 — Limite \(\beta\to\infty\), Wilson SU(2)

On suppose \(L\) multiple de 3, \(Q\) surjective et la linéarisation précisée dans la question. On note \(N_s=L^4\), \(N_\ell=4L^4\), \(N_p=6L^4\), \(N_c=(L/3)^4=L^4/81\), et \(C\) le rotationnel linéaire : \(A_p=CA\).

### (a) Point identité

Dans chaque couleur,
\[
\boxed{J(1)=Q,\qquad T_1F_1=\{A:QA=0\}=\ker Q.}
\]
Avec les trois couleurs, \(J(1)\) est \(Q\otimes I_3\) (à l’ordre des coordonnées près).

### (b) Gaussienne conditionnelle et KKT

Au premier ordre,
\[
S=\frac\beta2 A^TC^TCA+O(\beta A^4).
\]
Soit \(\mathcal H=\ker Q\), et \(\mathcal G_0\subset\mathcal H\) les transformations de jauge vérifiant \(g=1\) aux centres des blocs. Si \(P_0\) est le projecteur orthogonal sur \(\mathcal H\ominus\mathcal G_0\), la covariance physique par couleur est
\[
\boxed{
\Sigma_Q=\beta^{-1}(C^TC\big|_{\mathcal H\ominus\mathcal G_0})^{-1},\qquad
\operatorname{Cov}(A)=\beta^{-1}P_0
\left(C^TC+P_{\mathcal H^\perp}+P_{\mathcal G_0}\right)^{-1}P_0,}
\]
la seconde écriture utilisant les projecteurs sur les sous-espaces indiqués. Les directions de jauge sont fixées ou quotientées ; elles n’affectent pas les plaquettes.

Pour \(V=e^{i\bar y}\) proche de 1, le minimiseur sous contrainte est le KKT
\[
C^TCA_*+Q^T\lambda=0,\qquad QA_*=\bar y.
\]
Après une fixation complète rendant \(\widehat K>0\), cela donne
\[
A_*=\widehat K^{-1}Q^T
(Q\widehat K^{-1}Q^T)^{-1}\bar y,
\]
et les fluctuations autour de \(A_*\) ont la covariance \(\Sigma_Q\). C’est la version quadratique de \(\min\{S:QA=\bar y\}\).

### (c) Effet du facteur jacobien

L’inversion de tous les liens donne \(\bar U(U^{-1})=\bar U(U)^{-1}\). Les différentielles de l’inversion sont orthogonales dans les coordonnées tangentes, donc \(\det G(U^{-1})=\det G(U)\). Par conséquent
\[
\boxed{\log\det G(A)=\log\det G(0)+O(|A|^2),}
\]
sans terme linéaire. Comme \(A=O(\beta^{-1/2})\), la correction au poids est \(O(\beta^{-1})\). Elle ne change pas la limite de \(\beta\langle1-P\rangle\) ; elle modifie ce type d’observable à un ordre relatif \(O(\beta^{-1})\), soit un ordre absolu \(O(\beta^{-2})\).

### (d) Coefficient exact

Dans la gaussienne physique, \(C\Sigma_QC^T\) est \(\beta^{-1}\) fois un projecteur. Ainsi, pour la moyenne sur toutes les plaquettes,
\[
\lim_{\beta\to\infty}\beta\langle1-\tfrac12\operatorname{Tr}U_p\rangle
=\frac32\,\frac{\operatorname{rang}(C|_{\mathcal H})}{N_p}.
\]
Le rang par couleur est
\[
\dim\ker Q-\dim\mathcal G_0
=(4L^4-4N_c)-(L^4-N_c)=3(L^4-N_c).
\]
Donc
\[
\boxed{
\lim_{\beta\to\infty}\beta\langle1-\tfrac12\operatorname{Tr}U_p\rangle_{V=1}
=\frac32\frac{3L^4(1-3^{-4})}{6L^4}
=\frac{20}{27}=0.740740740741.}
\]

Pour comparaison, sur le tore périodique non conditionné, il faut aussi retrancher les **quatre modes harmoniques fermés non de jauge** (un par direction). Le rang de \(C\) par couleur est donc
\[
4L^4-(L^4-1)-4=3L^4-3,
\]
et
\[
\boxed{
\lim_{\beta\to\infty}\beta\langle1-P\rangle_{\rm non\ cond.}
=\frac32\frac{3L^4-3}{6L^4}
=\frac34-\frac3{4L^4}.}
\]
Pour une vraie moyenne grossière, \(QA=0\) élimine ces quatre modes harmoniques dans la fibre. À \(L=3\), les deux coefficients valent \(20/27\). Pour \(L>3\), la différence conditionné moins non conditionné est \(-1/108+3/(4L^4)\).

## Fichiers de vérification

- `verify_q2_q3.py`, sortie `verify_q2_q3.out` : Q2 et Q3, haute précision.
- `verify_q4_q5.py`, sortie `verify_q4_q5.out` : Bessel, quadratures indépendantes, identités de co-aire et valeurs numériques.
- `verify_q5_gaussian.py`, sortie `verify_q5_gaussian.out` : formules de moyenne et covariance, contrainte dure et pénalité.
- `verify_q6.py`, sortie `verify_q6.out` : construction explicite du rotationnel périodique \(L=3\), rangs et coefficients.
