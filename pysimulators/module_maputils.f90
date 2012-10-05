module maputils
    
  implicit none

  integer, parameter :: p = kind(1.d0)
  real(p), parameter ::                                                        &
  NaN  = transfer('1111111111111000000000000000000000000000000000000000000000000000'b, 0._p), &
  mInf = transfer('1111111111110000000000000000000000000000000000000000000000000000'b, 0._p), &
  pInf = transfer('0111111111110000000000000000000000000000000000000000000000000000'b, 0._p)


contains


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

end module maputils
