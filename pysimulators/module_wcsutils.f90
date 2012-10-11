module wcsutils

    implicit none

    integer, parameter :: p = selected_real_kind(15,307)
    real(p), parameter :: PI = 4_p * atan(1._p)
    real(p), parameter :: DEG2RAD = PI / 180._p
    real(p), parameter :: RAD2DEG = 180._p / PI
    real(p), parameter ::                                                                                                          &
        NaN  = transfer('1111111111111000000000000000000000000000000000000000000000000000'b, 0._p),                                &
        mInf = transfer('1111111111110000000000000000000000000000000000000000000000000000'b, 0._p),                                &
        pInf = transfer('0111111111110000000000000000000000000000000000000000000000000000'b, 0._p)

    type Astrometry
        integer          :: naxis(2)
        real(p)          :: crpix(2), crval(2), cd(2,2)
        character(len=8) :: ctype(2), cunit(2)
    end type Astrometry

contains

    subroutine angle_lonlat(lon1, lat1, m, lon2, lat2, n, angle)

        real(p), intent(in)  :: lon1(m), lat1(m)
        integer, intent(in)  :: m
        real(p), intent(in)  :: lon2(n), lat2(n)
        integer, intent(in)  :: n
        real(p), intent(out) :: angle(max(m,n))

        if (m == 1) then
            call angle_lonlat__inner(lon1(1), lat1(1), lon2, lat2, angle)
        else if (n == 1) then
            call angle_lonlat__inner(lon1, lat1, lon2(1), lat2(1), angle)
        else
            call angle_lonlat__inner(lon1, lat1, lon2, lat2, angle)
        end if

    end subroutine angle_lonlat


    !---------------------------------------------------------------------------


    subroutine barycenter_lonlat(lon, lat, n, lon0, lat0)

        integer*8, intent(in) :: n
        real*8, intent(in)    :: lon(n), lat(n)
        real*8, intent(out)   :: lon0, lat0

        real*8    :: x, y, z, phi, cotheta, cocotheta
        integer*8 :: i

        if (size(lon) == 0) then
            lon0 = NaN
            lat0 = NaN
            return
        end if

        x = 0
        y = 0
        z = 0
        !$omp parallel do reduction(+:x,y,z) private(phi,cotheta,cocotheta)
        do i = 1, n
            phi = lon(i) * DEG2RAD
            cotheta = lat(i) * DEG2RAD
            if (cotheta /= cotheta .or. phi /= phi) cycle
            cocotheta = cos(cotheta)
            x = x + cocotheta * cos(phi)
            y = y + cocotheta * sin(phi)
            z = z + sin(cotheta)
        end do
        !$omp end parallel do
        
        lon0 = modulo(atan2(y,x) * RAD2DEG + 360._p, 360._p)
        lat0 = acos(sqrt((x**2 + y**2)/(x**2 + y**2 + z**2))) * RAD2DEG
        lat0 = sign(lat0, z)
        
    end subroutine barycenter_lonlat


    !---------------------------------------------------------------------------


    subroutine create_grid_square(nx, ny, size, filling_factor, xreflection, yreflection, rotation, xcenter, ycenter, coords)
        !f2py threadsafe
        integer, intent(in)    :: nx                 ! number of detectors along the x axis
        integer, intent(in)    :: ny                 ! number of detectors along the y axis
        real*8, intent(in)     :: size               ! size of the detector placeholder
        real*8, intent(in)     :: filling_factor     ! fraction of transmitting detector area
        logical, intent(in)    :: xreflection        ! reflection along the x-axis (before rotation)
        logical, intent(in)    :: yreflection        ! reflection along the y-axis (before rotation)
        real*8, intent(in)     :: rotation           ! counter-clockwise rotation in degrees (before translation)
        real*8, intent(in)     :: xcenter, ycenter   ! coordinates of the grid center
        real*8, intent(inout)  :: coords(2,4,nx,ny)  ! output coordinates of the detector corners (first dimension is x and y)

        integer :: i, j, k
        real*8 :: x, y, x0, y0, size_eff, r11, r12, r21, r22, tmp

        size_eff = size * sqrt(filling_factor)
        r11 = cos(DEG2RAD * rotation)
        r21 = sin(DEG2RAD * rotation)
        r12 = -r21
        r22 = r11
        if (xreflection) then
            r11 = -r11
            r21 = -r21
        end if
        if (yreflection) then
            r12 = -r12
            r22 = -r22
        end if
        x0 = -0.5_p * ((nx + 1) * size + size_eff)
        y0 = -0.5_p * ((ny + 1) * size + size_eff)
        
        do j = 1, ny
            y = y0 + size * j
            do i = 1, nx 
                x = x0 + size * i
                coords(1,1,i,j) = x
                coords(2,1,i,j) = y
                coords(1,2,i,j) = x + size_eff
                coords(2,2,i,j) = y
                coords(1,3,i,j) = x + size_eff
                coords(2,3,i,j) = y + size_eff
                coords(1,4,i,j) = x
                coords(2,4,i,j) = y + size_eff
                do k = 1, 4
                    tmp = coords(1,k,i,j)
                    coords(1,k,i,j) = xcenter + r11 * coords(1,k,i,j) + r12 * coords(2,k,i,j)
                    coords(2,k,i,j) = ycenter + r21 * tmp             + r22 * coords(2,k,i,j)
                end do
            end do
        end do
        
    end subroutine create_grid_square


    !---------------------------------------------------------------------------
    
    
    subroutine mean_degrees(array, n, mean)

        integer*8, intent(in) :: n
        real(p), intent(in)   :: array(n)
        real(p), intent(out)  :: mean

        real(p) :: val
        integer :: isample, n180, nvalids
        logical :: zero_minus, zero_plus

        mean = 0
        zero_minus = .false.
        zero_plus  = .false.
        n180 = 0
        nvalids = 0

        !$omp parallel do default(shared) reduction(+:n180, nvalids, mean)     &
        !$omp reduction(.or.:zero_minus,zero_plus) private(isample, val)
        do isample = 1, n
            val = modulo(array(isample), 360._p)
            if (val /= val) cycle
            zero_minus = zero_minus .or. val > 270._p
            zero_plus  = zero_plus  .or. val <= 90._p
            if (val >= 180._p) n180 = n180 + 1
            mean = mean + val
            nvalids = nvalids + 1
        end do
        !$omp end parallel do

        if (zero_minus .and. zero_plus) mean = mean - 360._p * n180

        if (nvalids > 0) then
            mean = modulo(mean / nvalids, 360._p)
        else
            mean = NaN
        end if

    end subroutine mean_degrees


    !---------------------------------------------------------------------------


    subroutine minmax_degrees(array, n, minv, maxv)

        real(p), intent(in)   :: array(n)
        integer*8, intent(in) :: n
        real(p), intent(out)  :: minv, maxv

        real(p) :: val, meanv
        integer :: isample

        call mean_degrees(array, size(array, kind=8), meanv)
        if (meanv /= meanv) then
            minv = NaN
            maxv = NaN
            return
        end if

        minv = pInf
        maxv = mInf
        !$omp parallel do default(shared) reduction(min:minv) reduction(max:maxv) private(val)
        do isample=1, size(array)
            val = array(isample)
            if (val /= val) cycle
            val = modulo(val, 360._p)
            if (val > meanv) then
                if (abs(val-360._p-meanv) < abs(val-meanv)) val = val - 360._p
            else
                if (abs(val+360._p-meanv) < abs(val-meanv)) val = val + 360._p
            end if
            minv = min(minv, val)
            maxv = max(maxv, val)
        end do
        !$omp end parallel do

        minv = modulo(minv, 360._p)
        maxv = modulo(maxv, 360._p)

    end subroutine minmax_degrees


!-------------------------------------------------------------------------------


!!$subroutine projection_scale(header, nx, ny, array, status)
!!$
!!$    use module_wcs,     only : Astrometry, init_astrometry, projection_scale_ => projection_scale
!!$    implicit none
!!$
!!$    character(len=*), intent(in) :: header
!!$    integer, intent(in)          :: nx, ny
!!$    real(p), intent(out)         :: array(nx,ny)
!!$    integer, intent(out)         :: status
!!$
!!$    type(Astrometry) :: astr
!!$
!!$    call init_astrometry(header, astr, status)
!!$    if (status /= 0) return
!!$
!!$    call projection_scale_(astr, array, status)
!!$    if (status /= 0) return
!!$
!!$end subroutine projection_scale


    !---------------------------------------------------------------------------
    !---------------------------------------------------------------------------
    !---------------------------------------------------------------------------
    !---------------------------------------------------------------------------


    pure elemental subroutine angle_lonlat__inner(lon1, lat1, lon2, lat2, angle)

        real*8, intent(in)  :: lon1, lat1, lon2, lat2
        real*8, intent(out) :: angle
        
        angle = acos(cos(lat1*DEG2RAD) * cos(lat2*DEG2RAD) *                   &
                     cos((lon2 - lon1)*DEG2RAD) +                              &
                     sin(lat1*DEG2RAD) * sin(lat2*DEG2RAD)) * RAD2DEG

    end subroutine angle_lonlat__inner


end module wcsutils
