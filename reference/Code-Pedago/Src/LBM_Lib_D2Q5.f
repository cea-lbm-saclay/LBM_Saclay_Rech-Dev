c*******************************i*****************************************
c
      subroutine Calc_Moment0 (Moment0, f, nx, ny)
c
c---- Calcul du moment d'ordre 0 : concentration pour le transport
c     ou la densité pour le Navier-Stokes. Simple somme sur k des f_k
c
      implicit none
      integer i, j, k, nx, ny, ni
      parameter (ni=4)
      double precision Moment0(0:nx+1,0:ny+1), f(0:ni,0:nx+1,0:ny+1)

      do j = 0, ny+1
         do i = 0, nx+1
            Moment0 (i,j) = 0.d0
            do k = 0 , ni
               Moment0(i,j) = Moment0(i,j) + f(k,i,j)
            enddo
         enddo
      enddo
c
      end
c
c************************Calcul de g_equilibre*********************************
c
      subroutine Calcul_Feq_standard_D2Q5 (f_equ, phi, Vx, Vy,
     .                                     w1   , nx , ny, dx,
     .                                     dt   ) 

      implicit none 
      integer i, j, ni, nx, ny
      parameter (ni=4)
      double precision c, dx, dt
      double precision f_equ(0:ni,0:nx+1,0:ny+1), w1(0:ni)
      double precision Vx(0:nx+1,0:ny+1), Vy(0:nx+1,0:ny+1),
     .                 phi(0:nx+1,0:ny+1)
c
      c = dx/dt
      f_equ = 0.d0
      do j = 0 , ny+1
         do i = 0 , nx+1
            f_equ(0,i,j) = w1(0) * phi(i,j)
            f_equ(1,i,j) = w1(1) * phi(i,j)*(1.d0 + 3.d0*Vx(i,j)/c)
            f_equ(2,i,j) = w1(2) * phi(i,j)*(1.d0 + 3.d0*Vy(i,j)/c)
            f_equ(3,i,j) = w1(3) * phi(i,j)*(1.d0 - 3.d0*Vx(i,j)/c)
            f_equ(4,i,j) = w1(4) * phi(i,j)*(1.d0 - 3.d0*Vy(i,j)/c)
         enddo
      enddo
   
      end
c
c***************Conditions aux Limites******************************
c
      subroutine Conditions_Limites_flux_nul_D2Q5(f, nx, ny)
c
      implicit none
      integer nx, ny, ni, i, j
      parameter (ni=4)
      double precision f(0:ni,0:nx+1,0:ny+1)
c
c---- Limites en bas et en haut
c
      do i = 0, nx+1
         f(2,i,0   ) = f(4,i,0   )
         f(4,i,ny+1) = f(2,i,ny+1)
      enddo
c
c---- Limites à gauche et à droite
c
      do j = 0, ny+1
         f(1,0   ,j) = f(3,0   ,j)
         f(3,nx+1,j) = f(1,nx+1,j)
      enddo
      
c
      end
c
c************************* Collision BGK *****************************************
c
      subroutine CollisionBGK_D2Q5 (f_star, f, f_equ, nx, ny, ni, tau)
c
c---- Collision BGK
c
      implicit none
      double precision tau
      double precision f     (0:ni,0:nx+1,0:ny+1),
     .                 f_star(0:ni,0:nx+1,0:ny+1),
     .                 f_equ (0:ni,0:nx+1,0:ny+1)
      integer i, j, k, nx, ny, ni
     
      f_star = 0.d0
      do j = 0 , ny+1
         do i = 0 , nx+1
            do k = 0 , ni
               f_star(k,i,j) = f(k,i,j) -
     .                   (1.d0 / tau) * (f(k,i,j) - f_equ(k,i,j))  
            enddo
         enddo
      enddo
c
      f = f_star
c
      end
c 
c*************************  Déplacement *******************************************
c
      subroutine deplacement_d2q5 (f, nx, ny)

      implicit none
      integer i, j, ni , nx, ny
      parameter (ni=4)
      double precision f(0:ni,0:nx+1,0:ny+1)
c
      do j = 0, ny+1
         do i = nx+1, 1, -1
            f(1,i,j) = f(1,i-1,j)
         enddo
      enddo
c
c---- Direction e2
c
      do j = ny+1, 1,-1
         do i = 0, nx+1
            f(2,i,j) = f(2,i,j-1)
         enddo
      enddo
c
c---- Direction e3
c
      do j = 0, ny+1
         do i = 0, nx
            f(3,i,j) = f(3,i+1,j)
         enddo
      enddo
c
c---- Direction e4
c
      do j = 0, ny
         do i = 0, nx+1
            f(4,i,j) = f(4,i,j+1)
         enddo
      enddo
c
      end
