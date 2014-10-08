module datautils
    
  use module_tamasis, only : p
  use module_math_old,    only : NaN, distance_1d_ => distance_1d, distance_2d_ => distance_2d, distance_3d_ => distance_3d
  implicit none

contains

  subroutine distance_1d(distance, nx, origin, resolution)

      real(p), intent(out) :: distance(0:nx-1)
      integer, intent(in)  :: nx
      real(p), intent(in)  :: origin
      real(p), intent(in)  :: resolution

      call distance_1d_(distance, origin, resolution)

  end subroutine distance_1d


  !-----------------------------------------------------------------------------


  subroutine distance_2d(distance, nx, ny, origin, resolution)

      real(p), intent(out) :: distance(0:nx-1,0:ny-1)
      integer, intent(in)  :: nx, ny
      real(p), intent(in)  :: origin(2)
      real(p), intent(in)  :: resolution(2)

      call distance_2d_(distance, origin, resolution)

  end subroutine distance_2d


  !-----------------------------------------------------------------------------


  subroutine distance_3d(distance, nx, ny, nz, origin, resolution)

      real(p), intent(out) :: distance(0:nx-1,0:ny-1,0:nz-1)
      integer, intent(in)  :: nx, ny, nz
      real(p), intent(in)  :: origin(3)
      real(p), intent(in)  :: resolution(3)

      call distance_3d_(distance, origin, resolution)

  end subroutine distance_3d


  !-----------------------------------------------------------------------------


  subroutine profile_axisymmetric_2d(array, nx, ny, origin, bin, nbins, x, y, n)

      real*8, intent(in)   :: array(0:nx-1,0:ny-1)
      integer, intent(in)  :: nx, ny
      real*8, intent(in)   :: origin(2)
      real*8, intent(in)   :: bin
      integer, intent(in)  :: nbins
      real*8, intent(out)  :: x(nbins)
      real*8, intent(out)  :: y(nbins)
      integer, intent(out) :: n(nbins)

      integer :: i, j, ibin
      real*8  :: distance, val, tmpj

      x = 0
      y = 0
      n = 0
      !$omp parallel do private(distance, ibin, tmpj, val) reduction(+:x,y,n)
      do j = 0, ny - 1
          tmpj = (j - origin(2))**2
          do i = 0, nx - 1
              val = array(i,j)
              if (val /= val) cycle
              distance = sqrt((i - origin(1))**2 + tmpj)
              ibin = int(distance / bin) + 1
              if (ibin > nbins) cycle
              x(ibin) = x(ibin) + distance
              y(ibin) = y(ibin) + val
              n(ibin) = n(ibin) + 1
          end do
      end do
      !$omp end parallel do

      do i = 1, nbins
          if (n(i) /= 0) then
              x(i) = x(i) / n(i)
              y(i) = y(i) / n(i)
          else
              x(i) = bin * (i - 0.5_p)
              y(i) = NaN
          end if
      end do
 
  end subroutine profile_axisymmetric_2d

end module datautils
