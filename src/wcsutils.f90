module wcsutils

    use module_tamasis, only : p
    use module_math_old, only : PI, DEG2RAD, RAD2DEG, NaN, mInf, pInf
    use module_fitstools,      only : ft_read_keyword
    use module_pointingmatrix, only : PointingElement, xy2pmatrix, xy2roi, roi2pmatrix_cartesian
    use module_wcs,            only : ad2xy_gnomonic, ad2xys_gnomonic, init_astrometry
    implicit none

    integer, parameter :: NEAREST_NEIGHBOUR = 0
    integer, parameter :: SHARP_EDGES = 1

contains

    subroutine angle_lonlat(lon1, lat1, m, lon2, lat2, n, angle)
        use module_math_old, only : angle_lonlat_ => angle_lonlat

        real(p), intent(in)  :: lon1(m), lat1(m)
        integer, intent(in)  :: m
        real(p), intent(in)  :: lon2(n), lat2(n)
        integer, intent(in)  :: n
        real(p), intent(out) :: angle(max(m,n))

        if (m == 1) then
            call angle_lonlat_(lon1(1), lat1(1), lon2, lat2, angle)
        else if (n == 1) then
            call angle_lonlat_(lon1, lat1, lon2(1), lat2(1), angle)
        else
            call angle_lonlat_(lon1, lat1, lon2, lat2, angle)
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
    
    
    subroutine mean_degrees(array, n, mean)

        integer*8, intent(in) :: n
        real(p), intent(in)   :: array(n)
        real(p), intent(out)  :: mean

        real(p) :: val
        integer*8 :: isample, n180, nvalids
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

        integer*8, intent(in) :: n
        real(p), intent(in)   :: array(n)
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


    !---------------------------------------------------------------------------
    
    
    subroutine projection_scale(header, nx, ny, array, status)

        use module_wcs, only : Astrometry, init_astrometry, &
                               projection_scale_ => projection_scale
        character(len=*), intent(in) :: header
        integer, intent(in)          :: nx, ny
        real(p), intent(out)         :: array(nx,ny)
        integer, intent(out)         :: status

        type(Astrometry) :: astr

        call init_astrometry(header, astr, status)
        if (status /= 0) return

        call projection_scale_(astr, array, status)
        if (status /= 0) return

    end subroutine projection_scale


    !---------------------------------------------------------------------------


    subroutine object2ad(input, output, ncoords, ra, dec, pa)
        ! Convert coordinates in the instrument object plane in arc seconds into
        ! celestial coordinates in degrees, assuming a pointing direction and
        ! a position angle.
        !
        !         input(2) = Declination if PA=0
        !    \    |
        !     \PA_|
        !      \/ |
        !       \ |
        !        \|_______ input(1) = -R.A. if PA=0
        !
        ! 
        ! The routine is not accurate at the poles.

        integer*8, intent(in)    :: ncoords           ! number of coordinates
        real(p), intent(in)    :: input(2,ncoords)  ! input coordinates in instrument object plane
        real(p), intent(inout) :: output(2,ncoords) ! output in celestial coordinates
        real(p), intent(in)    :: ra, dec, pa       ! pointing direction is (0,0) in the local frame

        real(p) :: cospa, sinpa, c1, c2
        integer*8 :: i

        cospa = cos(pa * DEG2RAD)
        sinpa = sin(pa * DEG2RAD)

        do i = 1, ncoords
            c1 = input(1,i) / 3600._p
            c2 = input(2,i) / 3600._p
            output(2,i) = dec + (-c1 * sinpa + c2 * cospa)
            output(1,i) = ra  + ( c1 * cospa + c2 * sinpa) / cos(output(2,i) * DEG2RAD)
        end do

    end subroutine object2ad


    !---------------------------------------------------------------------------


    subroutine object2xy_minmax(coords, ncoords, ra, dec, pa, npointings, header, xmin, ymin, xmax, ymax, status)
        ! Return the minimum and maximum sky pixel coordinate values of coordinates in the instrument object plane.

        integer*8, intent(in)                      :: ncoords, npointings     ! #coordinates, #pointings
        real(p), intent(in)                        :: coords(2,ncoords)       ! instrument frame coordinates
        real(p), intent(in), dimension(npointings) :: ra, dec, pa             ! input pointings in celestial coordinates
        character(len=2880), intent(in)            :: header                  ! input FITS header
        real(p), intent(out)                       :: xmin, ymin, xmax, ymax  ! min and max values of the map coordinates
        integer, intent(out)                       :: status                  ! status flag

        real(p)   :: xy(2,ncoords)
        integer*8 :: ipointing

        call init_astrometry(header, status=status)
        if (status /= 0) return

        xmin = pInf
        xmax = mInf
        ymin = pInf
        ymax = mInf

#ifndef IFORT
        !$omp parallel do reduction(min:xmin,ymin) reduction(max:xmax,ymax) private(xy)
#endif
        do ipointing = 1, npointings

            call object2ad(coords, xy, ncoords, ra(ipointing), dec(ipointing), pa(ipointing))
            call ad2xy_gnomonic(xy(1,:), xy(2,:))
            xmin = min(xmin, minval(xy(1,:)))
            ymin = min(ymin, minval(xy(2,:)))
            xmax = max(xmax, maxval(xy(1,:)))
            ymax = max(ymax, maxval(xy(2,:)))

        end do
#ifndef IFORT
        !$omp end parallel do
#endif

    end subroutine object2xy_minmax


    !---------------------------------------------------------------------------


    subroutine object2pmatrix_nearest_neighbour(coords, ncoords, area, ra, dec, pa, masked, npointings, header, pmatrix, out,  &
                                                    status)
        !f2py integer*8, depend(npointings,ncoords) :: pmatrix(npointings*ncoords)
        integer*8, intent(in)                        :: ncoords, npointings     ! #coordinates, #pointings
        real(p), intent(in)                          :: coords(2,ncoords)       ! instrument frame coordinates
        real(p), intent(in)                          :: area(ncoords)           ! detector area / reference_area
        real(p), intent(in), dimension(npointings)   :: ra, dec, pa             ! input pointings in celestial coordinates
        logical*1, intent(in), dimension(npointings) :: masked                  ! pointing flags: true if masked, removed
        character(len=*), intent(in)                 :: header
        type(PointingElement), intent(inout)         :: pmatrix(1,npointings,ncoords)
        logical, intent(out)                         :: out
        integer, intent(out)                         :: status

        real(p)   :: coords2(2,ncoords), x(ncoords), y(ncoords), s(ncoords)
        integer*8 :: isample
        integer   :: nx, ny

        out = .false.
        call init_astrometry(header, status=status)
        if (status /= 0) return
        ! get the size of the map
        call ft_read_keyword(header, 'naxis1', nx, status=status)
        if (status /= 0) return
        call ft_read_keyword(header, 'naxis2', ny, status=status)
        if (status /= 0) return

        !$omp parallel do private(isample, coords2, x, y, s) reduction(.or. : out)
        ! loop over the samples which have not been removed
        do isample = 1, npointings

            if (masked(isample)) then
                pmatrix(1,isample,:)%index = -1
                pmatrix(1,isample,:)%value = 0
                cycle
            end if

            call object2ad(coords, coords2, ncoords, ra(isample), dec(isample), pa(isample))
            
            call ad2xys_gnomonic(coords2, x, y, s)

            ! the input map has flux densities, not surface brightness
            ! f_idetector = f_imap * weight
            ! with weight = detector_area / pixel_area
            ! and pixel_area = reference_area / s
            call xy2pmatrix(x, y, nx, ny, out, pmatrix(1,isample,:))
            pmatrix(1,isample,:)%value = real(s * area, kind=kind(pmatrix%value))

        end do
        !$omp end parallel do

    end subroutine object2pmatrix_nearest_neighbour


    !---------------------------------------------------------------------------


    subroutine object2pmatrix_sharp_edges(coords, nvertices, ncoords, ra, dec, pa, masked, npointings, header, pmatrix,        &
                                              npixels_per_sample, new_npixels_per_sample, out, status)
        !f2py integer*8, depend(npixels_per_sample,npointings,ncoords) :: pmatrix(npixels_per_sample*npointings*ncoords)
        integer*8, intent(in)                        :: nvertices, ncoords, npointings ! #vertices, #coordinates, #pointings
        real(p), intent(in)                          :: coords(2,nvertices,ncoords)   ! instrument frame coordinates
        real(p), intent(in), dimension(npointings)   :: ra, dec, pa         ! input pointings in celestial coordinates
        logical*1, intent(in), dimension(npointings) :: masked              ! pointing flags: true if masked, removed
        character(len=*), intent(in)                 :: header              ! sky map FITS header
        type(PointingElement), intent(inout)         :: pmatrix(npixels_per_sample,npointings,ncoords) ! the pointing matrix
        integer, intent(in)  :: npixels_per_sample     ! input maximum number of sky pixels intersected by a detector
        integer, intent(out) :: new_npixels_per_sample ! actual maximum number of sky pixels intersected by a detector
        logical, intent(out) :: out                    ! true if some coordinates fall outside of the map
        integer, intent(out) :: status

        real(p)   :: coords2(2,nvertices,ncoords)
        integer   :: roi(2,2,ncoords)
        integer*8 :: isample
        integer   :: nx, ny

        new_npixels_per_sample = 0
        out = .false.
        call init_astrometry(header, status=status)
        if (status /= 0) return
        ! get the size of the map
        call ft_read_keyword(header, 'naxis1', nx, status=status)
        if (status /= 0) return
        call ft_read_keyword(header, 'naxis2', ny, status=status)
        if (status /= 0) return

        !$omp parallel do private(isample, coords2, roi) &
        !$omp reduction(max : npixels_per_sample) reduction(.or. : out)
        do isample = 1, npointings

            if (masked(isample)) then
                pmatrix(:,isample,:)%index = -1
                pmatrix(:,isample,:)%value = 0
                cycle
            end if

            call object2ad(coords, coords2, ncoords*nvertices, ra(isample), dec(isample), pa(isample))
            call ad2xy_gnomonic(coords2(1,:,:), coords2(2,:,:))
            roi = xy2roi(coords2)
            call roi2pmatrix_cartesian(roi, coords2, nx, ny, new_npixels_per_sample, out, pmatrix(:,isample,:))

        end do
        !$omp end parallel do

    end subroutine object2pmatrix_sharp_edges


end module wcsutils
