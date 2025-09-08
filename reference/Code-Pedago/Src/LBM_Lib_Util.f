c
c**********c*********** **********************************
c
      subroutine Ecrit_Champ_Scal_VTK
     .                            (Champ2D, nx  , ny   ,
     .                             jpas   , name, char1,
     .                             chaine , dx  )
c
c---- Ecriture au format VTK pour visualisation PARAVIEW
c
      implicit none
      integer nx, ny, i, j, iou, jpas
      double precision Champ2D, dx
      dimension Champ2D(0:nx+1,0:ny+1)
      character tab*1, char7*7, name*8, char1*1, chaine*10
c
      tab = char(9)
      iou = 20
      call int2char7(char7,jpas)
c
      open (iou, file = name//trim(chaine)//char7//'.vtk',
     .                                           status = 'unknown')
      write(iou,1) nx+2, ny+2, 1, (nx+2)*(ny+2)
      do j = 0, ny+1
         do i = 0, nx+1
            write(iou,2) sngl(i*dx), sngl(j*dx), 0.0
         enddo
      enddo
      write(iou,3) (nx+2)*(ny+2), char1
      do j = 0, ny+1
         do i = 0, nx+1
            write(iou,5) sngl(Champ2D(i,j))
         enddo
      enddo
      close(iou)
c
 1    format ('# vtk DataFile Version 3.0',/,
     .        'Exemple STRUCTURED_GRID'   ,/,
     .        'ASCII'                     ,/,
     .        'DATASET STRUCTURED_GRID'   ,/,
     .        'DIMENSIONS ',3i6           ,/,
     .        'POINTS ',i8,' float')
 2    format (f7.2,1x,f7.2,1x,f7.2)
 3    format ('POINT_DATA ',i8       ,/,
     .        'SCALARS ',a1,' float' ,/,
     .        'LOOKUP_TABLE default' )
 5    format (1p,e13.5)
c
      end
c
c********1*********2*********3*********4*********5*********6*********7*********8
c
      subroutine int2char4 (char4,i)
c
c---- convertit un entier (0-999) en chaine de caractere
c
      implicit integer*4 (i-n)
c
      character*4 char4
c
      write (char4,'(i4)') i
      if (i .lt. 10) then
         char4(1:3) = '000'
      elseif (i .lt. 100) then
         char4(1:2) = '00'
      elseif (i .lt. 1000) then
         char4(1:1) = '0'
      endif
c
      return
      end
c
c********1*********2*********3*********4*********5*********6*********7*********8
c
      subroutine int2char7 (char7,i)
c
c---- convertit un entier (0-999999) en chaine de caractere
c
      implicit integer*4 (i-n)
c
      character*7 char7
c
      write (char7,'(i7)') i
      if (i .lt. 10) then
         char7(1:6) = '000000'
      elseif (i .lt. 100) then
         char7(1:5) = '00000'
      elseif (i .lt. 1000) then
         char7(1:4) = '0000'
      elseif (i .lt. 10000) then
         char7(1:3) = '000'
      elseif (i .lt. 100000) then
         char7(1:2) = '00'
      elseif (i .lt. 1000000) then
         char7(1:1) = '0'
      endif
c
      return
      end
c
c********1*********2*********3*********4*********5*********6*********7*********8
c
      subroutine GaussianHill2D_Init (ChampScal2D, C0, Sigma2,
     .                                X0         , Y0, dx    ,
     .                                nx         , ny)
c
c---- Mode 3D : CI pour les instabilites de Rayleigh-Taylor 3D
c
      implicit none
      integer i, j, nx, ny
      double precision X0, Y0    ,
     .                 pi, Sigma2, Ampl,
     .                 x , y     ,
     .                 dx, C0
      double precision ChampScal2D(0:nx+1, 0:ny+1)
c
      pi = dacos(-1.d0)
c
      Ampl = C0/(2.d0*pi*Sigma2)
      ChampScal2D = 0.d0
      do j = 0, ny+1
         y = j*dx
         do i = 0, nx+1
            x = i*dx
            ChampScal2D(i,j) = Ampl*
     .                              dexp(-((x-x0)**2 +
     .                                     (y-y0)**2 )/(2.d0*Sigma2))
         enddo
      enddo
c
      end
c
c********1*********2*********3*********4*********5*********6*********7*********8
c
      subroutine GaussianHill2D_Time (ChampScal2D, C0    , Sigma2,
     .                                X0         , Y0    , dx    ,
     .                                D_xx       , nx    , ny    ,
     .                                pastps     , dtime )
c
c---- Mode 3D : CI pour les instabilites de Rayleigh-Taylor 3D
c
      implicit none
      integer i     , j ,
     .        nx    , ny,
     .        pastps
      double precision X0      , Y0      ,
     .                 D_xx    ,
     .                 pi      , Sigma2  , Ampl    ,
     .                 x       , y       ,
     .                 dx      , C0      , temps   ,
     .                 Sigma_xx, dtime
      double precision ChampScal2D(0:nx+1, 0:ny+1)
c
      pi = dacos(-1.d0)
c
      temps = pastps*dtime
c
      Sigma_xx = Sigma2 + 2.d0*D_xx*temps
      Ampl = C0/(2.d0*pi*Sigma_xx)
      ChampScal2D = 0.d0
      do j = 0, ny+1
         y = j*dx
         do i = 0, nx+1
            x = i*dx
            ChampScal2D(i,j) = Ampl*
     .                              dexp(-((x-x0)**2 +
     .                                     (y-y0)**2)/(2.d0*Sigma_xx))
         enddo
      enddo
c
      end
