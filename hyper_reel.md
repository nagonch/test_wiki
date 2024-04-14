# HyperReel: High Fidelity 6 DoF Video with Ray Conditioned Sampling notes

[Paper link
](https://openaccess.thecvf.com/content/CVPR2023/papers/Attal_HyperReel_High-Fidelity_6-DoF_Video_With_Ray-Conditioned_Sampling_CVPR_2023_paper.pdf)
# Abstract

- **Volumetric scene representations** — basis of existing 6DoF videos techniques
- Underlying volume rendering techniques — trade of **quality***,* **speed of rendering and memory efficiency**
- Existing methods **fail to achieve real-time, small memory footprint and high quality**
- HyperReel — novel 6DoF video representation. Two core components
    - **Ray-conditioned** **sample predicting network
    - Compact and memory-efficient **volume represnetation**
- $18$ FPS at $1$MP resolution without custom CUDA code

# Introduction

- 6-DoF videos — free user exploration through changing head position and orientation
- Driven by **view synthesis**
- Modern methods: slow (*Neural 3D Video Synthesis from Multi-view Video***)**
- Speed-based methods are memory hungry (*Fourier PlenOctrees for dynamic radiance field rendering in real-time*)
- Recent approaches started considering dynamic scenes (*Streaming radiance fields for 3D video synthesis*)
- Modern approaches **struggle with reflections, refractions, view-dependant appearance**
- This method solves the problem of quality-memory-speed with **following ingredients:**
    - **Ray-conditioned sample prediction network**. Predicts sparse point samples and
        - Accelerates rendering
        - Improves quality for view-dependent scenes
    - **Memory-efficient dynamic volume representation**
        - Exploits spatio-temporal redundancy of a dynamic scene
        - Extend Tensorial Radiance Fields to represent a set of volumetric keyframes
- Contributions
    - ***Novel sample predicting network**.* Faster rendering, better at view-dependency
    - Memory-efficient **volume representation**
    - **6DoF video representation**, high resolution real-time rendering

# Related Work

### Novel View Synthesis

### Neural Radiance Fields

### Adaptive Sampling for Neural Volume Rendering

- **Reduce the number of times we query the network** to render a single ray
- NeRFs that **learn weights of each point along the ray** for real-time rendering
    - *DoNeRF*
    - *TermiNeRF*
    - *AdaNeRF*
- *AutoInt, DIVeR, Neural Light Fields* — single (ish) network evaluation per ray
- The network in this work: accelerates novel view synthesis and improves quality for challenging scenes

### 6–Degrees-of-Freedom Video

- Multi-view ****camera rigs** (360 field of view)
- **Multi-sphere images**

### 6-DoF from Multi-View Captures

- Utilize NeRFs
- **Directly extending volumetric representations** **can give high quality and speed, but **uses a lot of memory**
- Low storage at cost of speed/quality
    - *StreamRF*
    - *NeRFPlayer*

# Method

### NeRFs

- $F_{\theta} : (\boldsymbol{x}, \vec{\omega}) \rightarrow (L_e(\boldsymbol{x}, \vec{\omega}), \sigma(\boldsymbol{x}))$. $\theta$ — **trainable** weights
- **New viepoint** is generated as:
$C(\boldsymbol{o}, \vec{\omega}) = \int_{t_n}^{t^f} T(\boldsymbol{o}, \boldsymbol{x_t}) \sigma(\boldsymbol{x_t}) L_e(\boldsymbol{x_t}, \vec{\omega}) dt$
- **Descrete** **version approximation:
$C(\boldsymbol{o}, \vec{\omega}) \approx \sum_{k=1}^N \omega_k L_e(\boldsymbol{x_k}, \vec{\omega})$
- $w_k = \hat{T}(\boldsymbol{o}, \boldsymbol{x_k}) (1 - e^{-\sigma(\boldsymbol{x_k})\Delta \boldsymbol{x_k}})$ — **contribution**
of point $k$ to the output color
- $N$ MLP calls/samples. Larger $N$ — **better**
- Works for **static** scene

### Sample Networks for Volume Rendering

- We need to sample points with non-zero $w_k$ (non-zero contribution to the output color)
- Importance sampling is used to reduce the number of queries, but still requires $\sim 10^2$ samples
- **This work:** uses a network to predict
the contributing set of samples of a **particular ray** (*sample predicting network*)
$E_{\phi} : (\boldsymbol{o}, \vec{\omega}) \rightarrow (\boldsymbol{x_1}, ..., \boldsymbol{x_N})$
- Ray $(\boldsymbol{o}, \vec{\omega})$ is represented in either
**two plane parametrization**, or **Plucker coordinates**
**
- **Problem:** Predicted $(\boldsymbol{x_1}, ..., \boldsymbol{x_N})$ cannot be completely aribtrary,
otherwise it violates multi-view consistensy of renderings.
**Needs constraints!**
- Instead, for a ray $\boldsymbol{r} = (\boldsymbol{o}, \vec{\omega}) = (\vec{\omega}, \vec{\omega} \times \boldsymbol{o})$, a set of
**geometric primitives is predicted** $E_{\phi} : (\boldsymbol{o}, \vec{\omega}) \rightarrow (G_1, ..., G_N)$
- To get $(\boldsymbol{x_1}, ..., \boldsymbol{x_N})$, $\boldsymbol{r}$ is intersected with $(G_1, ..., G_N)$,
which is a **differentiable operation**
**
- In practice, every $G_i$ is either an axis-aligned
$z$-**plane** (**forward-facing scenes**),
or a **spheric shell** centered
at the origin (**all other scenes***)*
- Two distinct rays **observing the same point**
in the scene will
**share their primitives**

### Flexible Sampling for Challenging Appearance

- Output sampling coordinates need **additional flexibility**
- Per-sample point **offests** are introduced
- They allow network to **cheat** and **deviate from
the predicted primitives**
when **necessary** (*challenging view-dependent conditions*)
$(\boldsymbol{d_1}, ..., \boldsymbol{d_n}) = (\gamma(\delta_1) \boldsymbol{e_1}, ..., \gamma(\delta_n) \boldsymbol{e_n})$
$(\boldsymbol{x_1}, ..., \boldsymbol{x_n}) \leftarrow (\boldsymbol{x_1} + \boldsymbol{d_1}, ..., \boldsymbol{x_n} + \boldsymbol{d_n})$
- $\delta_i$ are initialized as **negative values**, so $\gamma$
(*sigmoid*) is initially close to zero (**no offsets**)
- Predicted bias $\boldsymbol{d}$ allows modeling
viewpoint-dependent warping of the samples (i.e. refractions)

![image](https://github.com/nagonch/test_wiki/assets/44123818/05a01c77-df5e-425d-950d-305946e4743c)

### Keyframe-Based Dynamic Volumes

- Sampling the volume efficiently — **done**
- How is the volume represented though?
- Used **NeRF** variations
    - **TensoRF** — for static scenes
    - **Keyframe-based TensoRF** — keyframe-based dynamic volume representation

### TensoRF representation

- **TensoRF** factorizes the 3D volume as a set of outer products

![image](https://github.com/nagonch/test_wiki/assets/44123818/59ebfe7e-d53f-423b-b835-d784c4bd3883)

- One tensor representation for **appearance** (*color*) through 
spherical harmonics coefficients

$A(\boldsymbol{x_k}) = \mathcal{B}_1(f_1(x_k, y_k) \odot g_1(z_k)) + \mathcal{B}_2(f_2(x_k, z_k) \odot g_2(y_k))$ 
$+ \mathcal{B}_3(f_3(z_k, y_k) \odot g_3(x_k))$
    - $f_i, g_i$ — vector-valued **functions**
    - $\mathcal{B}_i$ — transformations that map to **spherical harmonic coefficients**
    - Eventual color at $\boldsymbol{x_k}$ is a **function** of spherical coefficients
    $A(\boldsymbol{x_k})$ and viewing direction $\boldsymbol{\vec{\omega}}$
- One representation for **geometry** (*density*)

$\sigma(\boldsymbol{x_k}) = \boldsymbol{1}^T (\boldsymbol{h_1} (x_k, y_k) \cdot \boldsymbol{k}_1(z_k)) + \boldsymbol{1}^T (\boldsymbol{h_2} (x_k, z_k) \cdot \boldsymbol{k}_2(y_k))$
$+ \boldsymbol{1}^T (\boldsymbol{h_3} (z_k, y_k) \cdot \boldsymbol{k}_3(x_k))$
    - $h_i, k_i$ — vector-valued **functions**
    - $\sigma(\boldsymbol{x_k})$ — denstity at $\boldsymbol{x_k}$

### TensoRF representation

- We need to modify TensoRF to handle **dynamic scenes**
- Denote $\tau_i$ as the time step corresponding to the $i^{\mathrm{th}}$ keyframe

$A(\boldsymbol{x_k}, \tau_i) = \mathcal{B}_1(f_1(x_k, y_k) \odot g_1(z_k, \tau_i))$
 $+ \mathcal{B}_2(f_2(x_k, z_k) \odot g_2(y_k, \tau_i))$ 
$+ \mathcal{B}_3(f_3(z_k, y_k) \odot g_3(x_k, \tau_i))$

$\sigma(\boldsymbol{x_k}, \tau_i) = \boldsymbol{1}^T (\boldsymbol{h_1} (x_k, y_k) \cdot \boldsymbol{k}_1(z_k, \tau_i))$
 $+ \boldsymbol{1}^T (\boldsymbol{h_2} (x_k, z_k) \cdot \boldsymbol{k}_2(y_k, \tau_i))$
$+ \boldsymbol{1}^T (\boldsymbol{h_3} (z_k, y_k) \cdot \boldsymbol{k}_3(x_k, \tau_i))$
- So the change is: $g_j$ and $k_j$ now depend on time
- The set of *scene-representing vecotrs* **changes** with time,
the set of *scene representing-matrices* **stays the same**
- Matrices stay constnat — **spatio-temporal
redundancy**
- Vectors change — scene **evolving in time**

### Rendering from Keyframe-Based Volumes

- Combining **representaiton** and **sampling procedure**
- **Problem 1**. Sample predicting network should take time $\tau$ into account
- **Problem 2**. We want to sample at arbitrary $\tau$ (outside of keyframes $\tau_i$)
- **Solution**
    - **Sampling-predicting** **network now outputs
    **velocities of each sampling point** at each keyframe $\tau_i$
    - We can **linearly interpolate the positions** of the sampling points
    inbetween the timeframes:
    $\boldsymbol{x_k} \leftarrow \boldsymbol{x_k} + \boldsymbol{v_k} (\tau_i - \tau)$ 
    (*backward warp with scene flow field*)

![image](https://github.com/nagonch/test_wiki/assets/44123818/52c6564d-0f4a-4575-afec-a3ce1d3e8997)

### Optimization

$\mathcal{L} = \mathcal{L_{\mathrm{L2}}} + \omega_{\mathrm{L1}}\mathcal{L_{\mathrm{L1}}} + \omega_{\mathrm{TV}}\mathcal{L_{\mathrm{TV}}}$
$\mathcal{L_{\mathrm{L2}}} = \sum_{\boldsymbol{r}, \tau} ||C(\boldsymbol{r}, \tau) - C_{\mathrm{GT}}(\boldsymbol{r}, \tau)||$

- **Total variation and sparsity regularizaton** is
applied to tensor components
- Loss is summed over **all training rays** **and times**

# Experiments

### Implementation Details

- **Single** RTX $3090$ ($24$ GB)
- Implemented in **Torch**
- Sample Network
    - $6$ layers **MLP**
    - $256$ **hidden** units
    - **Leaky** ReLU
    - $32$ $z$-**planes** **OR** $32$ spherical **shells**
- **Keyframe-based** volume representation
    - Every $4$-th frame — **keyframe**
    - Videos split into $50$-frame **chunks**
    - Training time for
    each chunk — $\sim 1.5$ hours

### Static Scenes

![image](https://github.com/nagonch/test_wiki/assets/44123818/3e956e9a-41a3-4ee9-90d8-516ec146508c)

- **DoNeRF** dataset
    - $6$ synthetic sequences
    - $800 \times 800$ resolution

### Dynamic Scenes

![image](https://github.com/nagonch/test_wiki/assets/44123818/98357f66-97f3-48e5-aec2-fec8ab09b459)

- **Technicolor Dataset**
    - $4 \times 4$ camera rig
    - $2048 \times 1088$ pixel videos
- **Neural 3D Video Dataset**
    - $6$ indoor videos
    - $20$ cameras
    - $2704 \times 2028$ resolution (downsampled $\frac{1}{2}$)
    - $64$ $z$-planes per ray instead of $32$
- **Google LF videos**
    - Light Field videos
    - $46$-fisheye camera rig

![image](https://github.com/nagonch/test_wiki/assets/44123818/a4acc9a0-f274-4fea-b4a5-afccea3d2640)

### Ablation Studies

- **Number of Keyframes**
    - **Increasing** — more complex motion
    - Increasing **too much** — volume’s capacity is overdistributed
    - Every $4$ frames — optimal
- **Network size and Number of Primitives**
    - *Tiny* — ($4$ layers, $128$ hidden unjits, $16$ primitives, $18$ FPS)
    - *Small* — ($4$ layers, $256$ hidden units, $16$ primitives, $9$ FPS)
- **NO sample network**
    
    ![image](https://github.com/nagonch/test_wiki/assets/44123818/c152b909-aa6c-43b5-8d89-11ae86685b51)
    
    ![image](https://github.com/nagonch/test_wiki/assets/44123818/3f6c4231-dc69-481f-bf45-ad89449b2de6)
    
- **Point Offset Ablation**
    
    ![image](https://github.com/nagonch/test_wiki/assets/44123818/7907fe42-542e-493d-a772-b8858aa3bf91)
    

# Conclusion

- Novel **6-DoF video**
    - Ray-conditioned **sampling network**
    - Keyframe-based **volume representation**
- Limitations
    - Reduction of quality **outside of convex
    hull of training images**
    - Keyframe representation
    **cannot be streamed**
    - **Not enough FPS** for VR
    
    ![image](https://github.com/nagonch/test_wiki/assets/44123818/b10d9b1f-0833-49eb-9688-bd82e0c42a9b)
