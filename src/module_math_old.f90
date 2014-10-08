! Copyright 2010-2011 Pierre Chanial
! All rights reserved
!
module module_math_old

    use iso_fortran_env, only : ERROR_UNIT
    use module_tamasis,  only : p

    implicit none
    private

    public :: PI
    public :: DEG2RAD
    public :: RAD2DEG
    public :: mInf
    public :: pInf
    public :: NaN

    public :: distance_1d, distance_2d, distance_3d
    public :: mad
    public :: mean
    public :: angle_lonlat
    public :: median_copy
    public :: median_nocopy
    public :: moment
    public :: nint_down
    public :: nint_up
    public :: sigma_clipping
    public :: stddev
    public :: sum_kahan
    public :: norm_l1
    public :: norm_huber
    public :: norm2
    public :: normp
    public :: dot
    public :: swap
    public :: eq_real
    public :: neq_real
    public :: shift_fast
    public :: shift_medium

    interface sum_kahan
        module procedure sum_kahan_1d, sum_kahan_2d, sum_kahan_3d
    end interface

    real(p), parameter :: PI = 4._p * atan(1._p)
    real(p), parameter :: DEG2RAD = PI / 180._p
    real(p), parameter :: RAD2DEG = 180._p / PI
    !XXX should use ieee_arithmetic instead when gfortran implements it. So far, gfortran doesn't allow NaN, mInf, pInf conversion
    ! between different real kinds...
#if PRECISION_REAL == 4
    real(p), parameter ::                                                                                                          &
        NaN  = transfer('11111111110000000000000000000000'b, 0._p),                                                                &
        mInf = transfer('11111111100000000000000000000000'b, 0._p),                                                                &
        pInf = transfer('01111111100000000000000000000000'b, 0._p)
#elif PRECISION_REAL == 8
    real(p), parameter ::                                                                                                          &
        NaN  = transfer('1111111111111000000000000000000000000000000000000000000000000000'b, 0._p),                                &
        mInf = transfer('1111111111110000000000000000000000000000000000000000000000000000'b, 0._p),                                &
        pInf = transfer('0111111111110000000000000000000000000000000000000000000000000000'b, 0._p)
#elif PRECISION_REAL == 16
    real(16), parameter :: NaN  = 'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF'z
    real(16), parameter :: mInf = 'FFFF0000000000000000000000000000'z
    real(16), parameter :: pInf = '7FFF0000000000000000000000000000'z
#endif


contains


  subroutine distance_1d(distance, origin, resolution)

      real(p), intent(out) :: distance(0:)
      real(p), intent(in)  :: origin
      real(p), intent(in)  :: resolution

      integer :: i
      real(p) :: resolution_

      resolution_ = abs(resolution)

      !$omp parallel do
      do i = 0, size(distance) - 1
          distance(i) = resolution_ * abs(i - origin)
      end do
      !$omp end parallel do

  end subroutine distance_1d


  !-----------------------------------------------------------------------------


  subroutine distance_2d(distance, origin, resolution)

      real(p), intent(out) :: distance(0:,0:)
      real(p), intent(in)  :: origin(2)
      real(p), intent(in)  :: resolution(2)

      integer :: i, j
      real(p) :: tmpj

      !$omp parallel do private(tmpj)
      do j = 0, size(distance, 2) - 1
          tmpj = (resolution(2) * (j - origin(2)))**2
          do i = 0, size(distance, 1) - 1
              distance(i,j) = sqrt((resolution(1) * (i - origin(1)))**2 + tmpj)
          end do
      end do
      !$omp end parallel do

  end subroutine distance_2d


  !-----------------------------------------------------------------------------


  subroutine distance_3d(distance, origin, resolution)

      real(p), intent(out) :: distance(0:,0:,0:)
      real(p), intent(in)  :: origin(3)
      real(p), intent(in)  :: resolution(3)

      integer :: i, j, k
      real(p) :: tmpj, tmpk

      !$omp parallel do private(tmpj, tmpk)
      do k = 0, size(distance, 3) - 1
          tmpk = (resolution(3) * (k - origin(3)))**2
          do j = 0, size(distance, 2) - 1
              tmpj = (resolution(2) * (j - origin(2)))**2
              do i = 0, size(distance, 1) - 1
                  distance(i,j,k) = sqrt((resolution(1) * (i - origin(1)))**2 + tmpj + tmpk)
              end do
          end do
      end do
      !$omp end parallel do

  end subroutine distance_3d


    subroutine moment(input, mean, variance, skewness, kurtosis, stddev, meandev, mask)

        real(p), intent(in)            :: input(:)
        real(p), intent(out), optional :: mean, variance, skewness
        real(p), intent(out), optional :: kurtosis, stddev, meandev
        logical, intent(in), optional  :: mask(:)

        integer              :: nsamples
        real(p)              :: m, var, sdev
        real(p), allocatable :: residuals(:)

        nsamples = size(input)

        if (present(mask)) then
            if (size(mask) /= nsamples) then
                write (ERROR_UNIT,'(a,2(i0,a))'), 'Error: the mask has an incompatible dimension (', size(mask), ' instead of ',   &
                     size(input), ').'
                stop 1
            end if
            nsamples = nsamples - count(mask)
        end if

        ! compute the mean
        if (nsamples == 0) then
            m = NaN
        else
            m = sum_kahan(input, mask) / nsamples
        end if
        if (present(mean)) mean = m

        if (.not. present(meandev)  .and. .not. present(stddev)   .and.                                                            &
            .not. present(variance) .and. .not. present(skewness) .and.                                                            &
            .not. present(kurtosis)) return

        allocate(residuals(size(input)))
        residuals = input - m

        ! compute mean deviation
        if (present(meandev)) then
            if (nsamples == 0) then
                meandev = NaN
            else
                meandev = sum_kahan(abs(residuals), mask) / nsamples
            end if
        end if

        ! compute variance
        if (nsamples <= 1) then
            var = NaN
        else
            var = sum_kahan(residuals**2, mask) / (nsamples-1)
        end if
        if (present(variance)) variance = var

        ! compute standard deviation
        sdev = sqrt(var)
        if (present(stddev)) stddev = sdev

        ! compute skewness
        if (present(skewness)) then
            if (nsamples == 0 .or. sdev == 0) then
                skewness = NaN
            else
                skewness = sum_kahan(residuals**3, mask) / (nsamples * sdev**3)
            end if
        end if

        ! compute kurtosis
        if (present(kurtosis)) then
            if (nsamples == 0 .or. sdev == 0) then
                kurtosis = NaN
            else
                kurtosis = sum_kahan(residuals**4, mask) / (nsamples * sdev**4) - 3
            end if
        end if

        deallocate(residuals)

    end subroutine moment


    !-------------------------------------------------------------------------------------------------------------------------------


    function mean(input, mask)

        real(p)                      :: mean
        real(p), intent(in)          :: input(:)
        logical, intent(in),optional :: mask(:)

        call moment(input, mean, mask=mask)

    end function mean


    !-------------------------------------------------------------------------------------------------------------------------------


    function stddev(input, mask)

        real(p)                       :: stddev
        real(p), intent(in)           :: input(:)
        logical, intent(in), optional :: mask(:)

        call moment(input, stddev=stddev, mask=mask)

    end function stddev


    !-------------------------------------------------------------------------------------------------------------------------------


    function sum_kahan_1d(input, mask) result (sum)

        real(p), intent(in)           :: input(:)
        logical, intent(in), optional :: mask(:)
        real(p)                       :: sum, c, t, y

        integer                       :: i

        if (size(input) == 0) then
            sum = 0
            return
        end if

        if (present(mask)) then
            if (size(mask) /= size(input)) then
                write (ERROR_UNIT,'(a,2(i0,a))'), 'Error: the mask has an incompatible dimension (', size(mask), ' instead of ',   &
                     size(input), ').'
                stop 1
            end if
            do i = 1, size(input)
                if (.not. mask(i)) exit
            end do
            if (i > size(input)) then
                sum = 0
                return
            end if
        else
            i = 1
        end if

        sum = input(i)
        c = 0
        do i = i+1, size(input)
            if (present(mask)) then
                if (mask(i)) cycle
            end if
            y = input(i) - c
            t = sum + y
            c = (t - sum) - y
            sum = t
        end do

    end function sum_kahan_1d


    !-------------------------------------------------------------------------------------------------------------------------------


    function sum_kahan_2d(input) result (sum)

        real(p), intent(in) :: input(:,:)
        real(p)             :: sum, c, t, y

        integer             :: i

        if (size(input) == 0) then
            sum = 0
            return
        end if

        sum = sum_kahan_1d(input(:,1))
        c = 0
        do i = 2, size(input,2)
            y = sum_kahan_1d(input(:,i)) - c
            t = sum + y
            c = (t - sum) - y
            sum = t
        end do
    end function sum_kahan_2d


    !-------------------------------------------------------------------------------------------------------------------------------


    function sum_kahan_3d(input) result (sum)

        real(p), intent(in) :: input(:,:,:)
        real(p)             :: sum, c, t, y

        integer             :: i

        if (size(input) == 0) then
            sum = 0
            return
        end if

        sum = sum_kahan_2d(input(:,:,1))
        c = 0
        do i = 2, size(input,3)
            y = sum_kahan_2d(input(:,:,i)) - c
            t = sum + y
            c = (t - sum) - y
            sum = t
        end do
    end function sum_kahan_3d


    !-------------------------------------------------------------------------------------------------------------------------------


    function norm_l1(input) result (sum)

        real(p), intent(in) :: input(:)
        real(p)             :: sum

        real(p) :: c, t, y
        integer :: i

        if (size(input) == 0) then
            sum = 0
            return
        end if

        sum = abs(input(1))
        c = 0
        do i = 2, size(input)
            y = abs(input(i)) - c
            t = sum + y
            c = (t - sum) - y
            sum = t
        end do

    end function norm_l1


    !-------------------------------------------------------------------------------------------------------------------------------


    function norm_huber(input, delta) result (sum)

        real(p), intent(in) :: input(:)
        real(p), intent(in) :: delta
        real(p)             :: sum

        real(p) :: c, t, y, a, b
        integer :: i

        if (size(input) == 0) then
            sum = 0
            return
        end if

        a = 2 * delta
        b = -delta**2

        if (abs(input(1)) < delta) then
            sum = input(1) ** 2
        else
            sum = a * abs(input(1)) + b
        end if

        c = 0
        do i = 2, size(input)
            if (abs(input(i)) < delta) then
                y = input(i) ** 2 - c
            else
                y = a * abs(input(i)) + b - c
            end if
            t = sum + y
            c = (t - sum) - y
            sum = t
        end do

    end function norm_huber


    !-------------------------------------------------------------------------------------------------------------------------------


    function norm2(input) result (sum)

        real(p), intent(in) :: input(:)
        real(p)             :: sum

        real(p) :: c, t, y
        integer :: i

        if (size(input) == 0) then
            sum = 0
            return
        end if

        sum = input(1)*input(1)
        c = 0
        do i = 2, size(input)
            y = input(i)*input(i) - c
            t = sum + y
            c = (t - sum) - y
            sum = t
        end do

    end function norm2


    !-------------------------------------------------------------------------------------------------------------------------------


    function normp(input, lp) result (sum)

        real(p), intent(in) :: input(:)
        real(p), intent(in) :: lp
        real(p)             :: sum

        real(p) :: c, t, y
        integer :: i

        if (size(input) == 0) then
            sum = 0
            return
        end if

        sum = abs(input(1))**lp
        c = 0
        do i = 2, size(input)
            y = abs(input(i))**lp - c
            t = sum + y
            c = (t - sum) - y
            sum = t
        end do

    end function normp


    !-------------------------------------------------------------------------------------------------------------------------------


    function dot(input1, input2) result (sum)

        real(p), intent(in) :: input1(:)
        real(p), intent(in) :: input2(:)
        real(p)             :: sum

        real(p) :: c, t, y
        integer :: i

        if (size(input1) == 0) then
            sum = 0
            return
        end if

        sum = input1(1)*input2(1)
        c = 0
        do i = 2, size(input1)
            y = input1(i)*input2(i) - c
            t = sum + y
            c = (t - sum) - y
            sum = t
        end do

    end function dot


    !-------------------------------------------------------------------------------------------------------------------------------


    pure elemental subroutine angle_lonlat(lon1, lat1, lon2, lat2, angle)

        real(p), intent(in)  :: lon1, lat1, lon2, lat2
        real(p), intent(out) :: angle
        
        angle = acos(cos(lat1*DEG2RAD) * cos(lat2*DEG2RAD) * cos((lon2 - lon1)*DEG2RAD) + sin(lat1*DEG2RAD) * sin(lat2*DEG2RAD)) * &
                RAD2DEG

    end subroutine angle_lonlat


    !-------------------------------------------------------------------------------------------------------------------------------

    ! remove me
    elemental function nint_down(x)

        integer             :: nint_down
        real(p), intent(in) :: x

        nint_down = nint(x)
        if (x > 0 .and. abs(x-nint_down) == 0.5_p) then
            nint_down = nint_down - 1
        end if

    end function nint_down


    !-------------------------------------------------------------------------------------------------------------------------------


    ! remove me
    elemental function nint_up(x)

        integer             :: nint_up
        real(p), intent(in) :: x

        nint_up = nint(x)
        if (x < 0 .and. abs(x-nint_up) == 0.5_p) then
            nint_up = nint_up + 1
        end if

    end function nint_up


    !-------------------------------------------------------------------------------------------------------------------------------


    elemental subroutine swap(a,b)

        real(p), intent(inout) :: a, b

        real(p)                :: tmp

        tmp = a
        a   = b
        b   = tmp

    end subroutine swap


    !-------------------------------------------------------------------------------------------------------------------------------


    function median_copy(array, remove_nan, mask) result (median) 

        real(p)             :: median
        real(p), intent(in) :: array(:)
        logical, intent(in) :: remove_nan
        logical*1, intent(in), optional :: mask(size(array))

        real(p), allocatable :: array_copy(:)

#ifdef IFORT
        if (remove_nan .and. present(mask)) then
            allocate (array_copy(count(array == array .and. .not. mask)))
        else if (present(mask)) then
            allocate (array_copy(count(.not. mask)))
        else if (remove_nan) then
            allocate (array_copy(count(array == array)))
        end if
#endif
        if (remove_nan .and. present(mask)) then
            array_copy = pack(array, array == array .and. .not. mask)
        else if (present(mask)) then
            array_copy = pack(array, .not. mask)
        else if (remove_nan) then
            array_copy = pack(array, array == array)
        else
            allocate (array_copy(size(array)))
            array_copy = array
        end if

        median = median_nocopy(array_copy, .false.)

    end function median_copy


    !-------------------------------------------------------------------------------------------------------------------------------


    ! This Quickselect routine is based on the algorithm described in
    ! "Numerical recipes in C", Second Edition,
    ! Cambridge University Press, 1992, Section 8.5, ISBN 0-521-43108-5
    ! input array may be reordered
    ! This code by Nicolas Devillard - 1998. Public domain.
    function median_nocopy(arr, remove_nan, mask) result (median)

        real(p)                :: median
        real(p), intent(inout) :: arr(0:)
        logical, intent(in)    :: remove_nan
        logical*1, intent(inout), optional :: mask(0:size(arr)-1)

        integer :: low, high, imedian, middle, ll, hh

        low = 0
        high = size(arr)-1

        if (remove_nan .or. present(mask)) then
            ! remove NaN or masked elements from input array
            ll = 0
            do
                if (ll > high) exit
                if (remove_nan .and. arr(ll) /= arr(ll)) go to 99
                if (present(mask)) then
                    if (mask(ll)) go to 99
                end if
                ! element is ok, let's check the next one
                ll = ll + 1
                cycle
                ! we remove the element by copying the last one into it
99              arr(ll) = arr(high)
                if (present(mask)) then
                    mask(ll) = mask(high)
                end if
                high = high - 1
            end do
        end if

        if (high < 0) then
            median = NaN
            return
        end if

        imedian = (low + high) / 2
        do
            if (high <= low) then
                median = arr(imedian)
                return
            end if

            if (high == low + 1) then  ! Two elements only
                if (arr(low) > arr(high)) call swap(arr(low), arr(high))
                median = arr(imedian)
                return
            end if

            ! Find imedian of low, middle and high items swap into position low
            middle = (low + high) / 2
            if (arr(middle) > arr(high)) call swap(arr(middle), arr(high))
            if (arr(low)    > arr(high)) call swap(arr(low),    arr(high))
            if (arr(middle) > arr(low))  call swap(arr(middle), arr(low))

            ! Swap low item (now in position middle) into position (low+1)
            call swap(arr(middle), arr(low+1)) 

            ! Nibble from each end towards middle, swapping items when stuck
            ll = low + 1
            hh = high
            do
                do 
                    ll = ll + 1
                    if (arr(low) <= arr(ll)) exit
                end do

                do 
                    hh = hh - 1
                    if (arr(hh)  <= arr(low)) exit
                end do

                if (hh < ll) exit

                call swap(arr(ll), arr(hh)) 

            end do

            ! Swap middle item (in position low) back into correct position
            call swap(arr(low), arr(hh)) 

            ! Re-set active partition
            if (hh <= imedian) low = ll
            if (hh >= imedian) high = hh - 1

        end do

    end function median_nocopy


    !-------------------------------------------------------------------------------------------------------------------------------


    ! returns the median absolute deviation
    function mad(x, m)

        real(p)                        :: mad
        real(p), intent(in)            :: x(:)
        real(p), intent(out), optional :: m

        real(p)                        :: x_(size(x)), med

        x_ = x
        med = median_nocopy(x_, .true.)
        x_ = abs(x_ - med)
        mad = median_nocopy(x_, .true.)

        if (present(m)) m = med

    end function mad


    !-------------------------------------------------------------------------------------------------------------------------------


    ! sigma clip an input vector. In the output mask, .true. means rejected
    subroutine sigma_clipping(input, mask, nsigma, nitermax)

        real(p), intent(in)           :: input(:)
        logical, intent(out)          :: mask(size(input))
        real(p), intent(in)           :: nsigma
        integer, intent(in), optional :: nitermax

        integer :: iter, nitermax_
        real(p) :: mean, stddev
        logical :: newmask(size(input))

        if (present(nitermax)) then
            nitermax_ = nitermax
        else
            nitermax_ = 0
        end if

        mask = .false.
        iter = 1
        do

            call moment(input, mean, stddev=stddev, mask=mask)
            newmask = mask .or. abs(input - mean) > nsigma * stddev
            if (count(newmask) == count(mask)) exit
            mask = newmask
            if (iter == nitermax_) exit

        end do

    end subroutine sigma_clipping


    !-------------------------------------------------------------------------------------------------------------------------------


    elemental function eq_real(a, b, rtol)

        logical                       :: eq_real
        real(p), intent(in)           :: a, b
        real(p), intent(in), optional :: rtol

        real(p) :: rtol_

        ! check for NaN values
        if (a /= a) then
            eq_real = b /= b
            return
        end if
        if (b /= b) then
            eq_real = .false.
            return
        end if

        if (present(rtol)) then
            rtol_ = rtol
        else
            rtol_ = 10._p * epsilon(1._p)
        end if

        eq_real = abs(a-b) <= rtol_ * max(abs(a),abs(b))

    end function eq_real


    !-------------------------------------------------------------------------------------------------------------------------------


    elemental function neq_real(a, b, rtol)

        logical                       :: neq_real
        real(p), intent(in)           :: a, b
        real(p), intent(in), optional :: rtol

        neq_real = .not. eq_real(a, b, rtol)

    end function neq_real


    !-------------------------------------------------------------------------------------------------------------------------------


    subroutine shift_fast(input, output, m, n, offset)

        integer*8, intent(in)  :: m, n
        real(p), intent(in)    :: input(m,n)
        real(p), intent(inout) :: output(m,n)
        integer*8, intent(in)  :: offset(:)

        integer*8 :: i, j, d

        !$omp parallel do private(i,j,d)
        do j = 1, n
            d = offset((j-1) / (n / size(offset)) + 1)
            if (d == 0) then
                output(:,j) = input(:,j)
                cycle
            end if
            if (d > 0) then
                do i = m, d+1, -1
                    output(i,j) = input(i-d,j)
                end do
                output(1:min(d,m),j) = 0
            else
                do i = 1, m+d
                    output(i,j) = input(i-d,j)
                end do
                output(max(m+d+1,1):m,j) = 0             
            end if
        end do
        !$omp end parallel do

    end subroutine shift_fast


    !-------------------------------------------------------------------------------------------------------------------------------


    subroutine shift_medium(input, output, m, n, o, offset)

        integer*8, intent(in)  :: m, n, o
        real(p), intent(in)    :: input(m,n,o)
        real(p), intent(inout) :: output(m,n,o)
        integer*8, intent(in)  :: offset(:)

        integer*8 :: j, k, d

        !$omp parallel do private(d)
        do k = 1, o
            d = offset((k-1) / (o / size(offset)) + 1)
            if (d == 0)  then
                output(:,:,k) = input(:,:,k)
                cycle
            end if
            if (d > 0) then
                do j = n, d+1, -1
                    output(:,j,k) = input(:,j-d,k)
                end do
                output(:,1:min(d,n),k) = 0
            else
                do j = 1, n+d
                    output(:,j,k) = input(:,j-d,k)
                end do
                output(:,max(n+d+1,1):n,k) = 0             
            end if
        end do
        !$omp end parallel do

    end subroutine shift_medium


end module module_math_old
