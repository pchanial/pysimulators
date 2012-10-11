module datautils
    
  use module_tamasis, only : p
  use module_math,    only : NaN
  implicit none

contains

  subroutine distance_1d(distance, nx, origin, resolution)

      integer, intent(in)  :: nx
      real(p), intent(out) :: distance(nx)
      real(p), intent(in)  :: origin
      real(p), intent(in)  :: resolution

      integer :: i
      real(p) :: resolution_

      resolution_ = abs(resolution)

      !$omp parallel do
      do i = 1, size(distance,1)
          distance(i) = resolution_ * abs(i - origin)
      end do
      !$omp end parallel do

  end subroutine distance_1d


  !-----------------------------------------------------------------------------


  subroutine distance_2d(distance, nx, ny, origin, resolution)

      integer, intent(in)  :: nx, ny
      real(p), intent(out) :: distance(nx,ny)
      real(p), intent(in)  :: origin(2)
      real(p), intent(in)  :: resolution(2)

      integer :: i, j
      real(p) :: tmpj

      !$omp parallel do private(tmpj)
      do j = 1, size(distance,2)
          tmpj = (resolution(2) * (j - origin(2)))**2
          do i = 1, size(distance,1)
              distance(i,j) = sqrt((resolution(1) * (i - origin(1)))**2 + tmpj)
          end do
      end do
      !$omp end parallel do

  end subroutine distance_2d


  !-----------------------------------------------------------------------------


  subroutine distance_3d(distance, nx, ny, nz, origin, resolution)

      integer, intent(in)  :: nx, ny, nz
      real(p), intent(out) :: distance(nx,ny,nz)
      real(p), intent(in)  :: origin(3)
      real(p), intent(in)  :: resolution(3)

      integer :: i, j, k
      real(p) :: tmpj, tmpk

      !$omp parallel do private(tmpj, tmpk)
      do k = 1, size(distance,3)
          tmpk = (resolution(3) * (k - origin(3)))**2
          do j = 1, size(distance,2)
              tmpj = (resolution(2) * (j - origin(2)))**2
              do i = 1, size(distance,1)
                  distance(i,j,k) = sqrt((resolution(1) * (i - origin(1)))**2 + tmpj + tmpk)
              end do
          end do
      end do
      !$omp end parallel do

  end subroutine distance_3d


  !---------------------------------------------------------------------------


  subroutine profile_axisymmetric_2d(array, nx, ny, origin, bin, nbins, x, y, n)

      real*8, intent(in)   :: array(nx,ny)
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
      do j = 1, ny
          tmpj = (j - origin(2))**2
          do i = 1, nx
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
