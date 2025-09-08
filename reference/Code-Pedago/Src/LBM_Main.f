      module MAIN_Boltzmann

      character*3 Opt_Fluide2D   , Opt_Transport2D ,
     .            Opt_Thermique2D, Opt_PhaseField2D
      character*4 Opt_Lattice_Fluide2D   , Opt_Lattice_Transport2D ,
     .            Opt_Lattice_Thermique2D, Opt_Lattice_PhaseField2D
      integer ni_phasefield, ni_thermique, ni_transport, ni_fluide
      character*8 Opt_MethNum_PhaseField2D, Opt_MethNum_Transport2D,
     .             Opt_Couplage_PhaseField2D_Fluides2D
c
      end module
c**********************************************************************
!     Déclaration des variables de transport

      module VariablesTransport
    
      integer nx , ny , nt , ni, nplot, t 
      doubleprecision  D, tau, dx, dt, time, C_init,
     .                 ux_init, uy_init
      doubleprecision, allocatable :: concent(:,:) , w1(:),
     .               f(:,:,:) , f_star(:,:,:) , f_equ(:,:,:), 
     .               Vx(:,:),Vy(:,:), Work(:,:)

      character*3 Opt_SolidDirect, Opt_Model_Transport, Solu_Gaussienne
      character*8 Approx_Integrales

      end module
c
c*****************************************************************************!     
c

      program Boltzman
c
      use MAIN_Boltzmann
      use VariablesTransport

      implicit none
c
c---- Lecture des donnees d'entrees
c
      call input
c
c---- Allocation des tableaux et conditions initiales
c
      call Initialisations
c
c---- Iterations en temps
c
      do t = 1 , nt
         time = t * dt
         call Resout_Eq_Transport_D2Q5
c
      enddo
c     
      call Deallocate_tab

      end

c********1*********2*********3*********4*********5*********6*********7*********8
c
      subroutine Initialisations
c
c
      use VariablesTransport
      use MAIN_Boltzmann

      implicit none
c
c---- Allocation des tableaux
c
      allocate (Concent(0:nx+1,0:ny+1), work(0:nx+1,0:ny+1))
      allocate (Vx(0:nx+1,0:ny+1), Vy(0:nx+1,0:ny+1))
      Concent = 0.d0 ; Vx = 0.d0 ; Vy = 0.d0
c
c---- Nb de directions de déplacement en D2Q5 : ni=4
c
      ni = 4
c
      allocate (f     (0:ni,0:nx+1,0:ny+1),f_star(0:ni,0:nx+1,0:ny+1),
     .          f_equ (0:ni,0:nx+1,0:ny+1))
      allocate (w1(0:ni))
      f_equ = 0.d0 ; f_star = 0.d0 ; f = 0.d0
c
c---- Définition des poids w du réseau
c
      call Definition_Poids_D2Q5
c
c---- Conditions initiales sur le transport et
c     initialisation du tau de collision
c
      call Initialise_Transport
c
      end
c
c**********************Initialisation ******************************************
c
      subroutine Initialise_Transport
c
c
c 
      use VariablesTransport
      use MAIN_Boltzmann
     

      implicit none
      double precision cinit, Sigma2, x0, y0
c
c---- Initialisation des vitesses
c
      Vx = ux_init ; Vy = uy_init
c
c---- Condition initiale
c
      cinit  = 5.d0
      Sigma2 = 1.d-2
      x0     = 1.d0
      y0     = 1.d0
c
c---- Condition intiale pour le LBM sous la forme d'une gaussienne
c
      call GaussianHill2D_Init (concent, cinit, Sigma2,
     .                          X0     , Y0   , dx    ,
     .                          nx     , ny   )
c
      call Ecrit_Champ_Scal_VTK(concent, nx   , ny   ,
     .                          0, 'LBM_D2Q5', 'C',
     .                          'Transport_', dx)
c
c---- Ecriture de la solution analytique au temps T si demandé
c
      if (Solu_Gaussienne == 'oui') then
         call GaussianHill2D_Time (work, cinit, Sigma2,
     .                             X0  , Y0   , dx    ,
     .                             D   , nx   , ny    ,
     .                             500 , dt   )
         call Ecrit_Champ_Scal_VTK(work, nx   , ny   ,
     .                             500, 'Gaussian', 'C',
     .                             'Transport_', dx)
      endif
c
c---- Ecriture à l'écran des valeurs pour vérifs
c
      write(6,*) 'dx =', dx, '   dt =', dt
      write(6,*) 'D  =', D

      tau  = 0.5d0 + (3.d0 * D * dt / (dx * dx))
      write(6,*) 'tau =', tau
c
c---- Initialisation de la F_eq avec C initialisé ci-dessus
c
      call Calcul_Feq_standard_D2Q5 (f_equ, Concent, Vx, Vy,
     .                               w1   , nx     , ny, dx,
     .                               dt   ) 

      f = f_equ
      
      end
c
c********1*********2*********3*********4*********5*********6*********7*********8
c
      subroutine Definition_Poids_D2Q5
c
c---- Définition des poids pour la fonction standart en D2Q5
c     Vérifié
c
      use VariablesTransport
      implicit none
      integer k
c
      w1(0) = 1.d0 / 3.d0

      do k = 1, 4
         w1(k) = 1.d0 / 6.d0 
      enddo
c
      end

c
c********1*********2*********3*********4*********5*********6*********7*********8
c
      subroutine Deallocate_Tab
c
      use VariablesTransport
c
      deallocate (concent, Vx, Vy, Work)
      deallocate (w1)
      deallocate (f, f_star, f_equ)
c
      end
