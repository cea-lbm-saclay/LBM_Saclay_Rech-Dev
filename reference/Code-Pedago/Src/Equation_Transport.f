c
c**********c*********** Transport D2Q5 **********************************
c
      subroutine Resout_Eq_Transport_D2Q5

      use VariablesTransport
      use MAIN_Boltzmann

      implicit none
c
c---- Fonction à l'équilibre pour l'ADE connaissant C et Ux et Uy
c
      call Calcul_Feq_standard_D2Q5 (f_equ, Concent, Vx, Vy,
     .                               w1   , nx     , ny, dx,
     .                               dt   )
c
c---- Collision connaissant Feq
c
      call CollisionBGK_D2Q5 (f_star, f, f_equ, nx, ny, ni, tau)
c
c---- Déplacement après la collision
c
      call deplacement_D2Q5 (f, nx, ny)
c
c---- Mise à Jour des composantes inconnues sur les limites après
c     l'étape de déplacement
c
      call Conditions_Limites_flux_nul_D2Q5(f, nx, ny)
c
c---- Calcul de la concentration
c
      call Calc_Moment0 (concent, f, nx, ny)
c
c---- Ecriture en VTK pour Paraview
c
      if (mod(t,nplot) == 0) then
         write(6,*) 'Ecriture au pas de temps ', t
         call Ecrit_Champ_Scal_VTK (Concent , nx        , ny ,
     .                              t       , 'LBM_D2Q5', 'C', 
     .                              'Transport_', dx)
      endif
c
      end
