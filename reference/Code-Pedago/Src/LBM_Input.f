 
      subroutine input

      use MAIN_Boltzmann
      use VariablesTransport
      implicit none
      integer iin, i
      character*1 shar1

      iin=10

      open (iin,file='data.in',status='old')
      do i = 1, 4
         read (iin,*) shar1
      enddo
      read (iin,*) Opt_Fluide2D    , Opt_Transport2D, Opt_Thermique2D,
     .             Opt_PhaseField2D, Opt_Couplage_PhaseField2D_Fluides2D
      read (iin,*) shar1
      read (iin,*) Opt_Lattice_Fluide2D   , Opt_Lattice_Transport2D ,
     .             Opt_Lattice_Thermique2D, Opt_Lattice_PhaseField2D

      do i = 1, 4
         read (iin,*) shar1
      enddo


c     5,100,50,100
      read (iin,*) nx, ny, nt, nplot
      read (iin,*) shar1
      read (iin,*) shar1
      read (iin,*) dx, dt

      do i = 1, 4
         read (iin,*) shar1
      enddo
      read (iin,*) D
      read (iin,*) shar1
      read (iin,*) C_init, ux_init, uy_init

      close(iin)

      end
