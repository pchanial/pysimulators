module wcsutils

    use module_tamasis, only : p
    use module_math,    only : PI, DEG2RAD, RAD2DEG, NaN, mInf, pInf
    use module_fitstools,      only : ft_read_keyword
    use module_pointingmatrix, only : PointingElement, xy2pmatrix, xy2roi, roi2pmatrix
    use module_projection,     only : convex_hull
    use module_wcs,            only : ad2xy_gnomonic, ad2xys_gnomonic, init_astrometry
    implicit none

    integer, parameter :: NEAREST_NEIGHBOUR = 0
    integer, parameter :: SHARP_EDGES = 1

contains

    subroutine angle_lonlat(lon1, lat1, m, lon2, lat2, n, angle)
        use module_math, only : angle_lonlat_ => angle_lonlat

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


    subroutine instrument2ad(input, output, ncoords, ra, dec, pa)
        ! Convert coordinates in the instrument frame (focal plane) into celestial coordinates,
        ! assuming a pointing direction and a position angle.
        ! The routine is not accurate at the poles.
        ! Input units are in arc seconds, output units in degrees.

        integer, intent(in)    :: ncoords           ! number of coordinates
        real(p), intent(in)    :: input(2,ncoords)  ! input coordinates in instrument frame
        real(p), intent(inout) :: output(2,ncoords) ! output in celestial coordinates
        real(p), intent(in)    :: ra, dec, pa       ! pointing direction is (0,0) in the local frame

        real(p) :: cospa, sinpa, c1, c2
        integer :: i

        cospa = cos(pa * DEG2RAD)
        sinpa = sin(pa * DEG2RAD)

        do i = 1, ncoords
            c1 = input(1,i) / 3600._p
            c2 = input(2,i) / 3600._p
            output(2,i) = dec + (-c1 * sinpa + c2 * cospa)
            output(1,i) = ra  + ( c1 * cospa + c2 * sinpa) / cos(output(2,i) * DEG2RAD)
        end do

    end subroutine instrument2ad


    !---------------------------------------------------------------------------


    subroutine instrument2xy_minmax(coords, ncoords, ra, dec, pa, npointings, header, xmin, ymin, xmax, ymax, status)
        ! Return the minimum and maximum sky pixel coordinate values of coordinates in the instrument frame.

        integer*8, intent(in)                      :: ncoords, npointings     ! #coordinates, #pointings
        real(p), intent(in)                        :: coords(2,ncoords)       ! instrument frame coordinates
        real(p), intent(in), dimension(npointings) :: ra, dec, pa             ! input pointings in celestial coordinates
        character(len=2880), intent(in)            :: header                  ! input FITS header
        real(p), intent(out)                       :: xmin, ymin, xmax, ymax  ! min and max values of the map coordinates
        integer, intent(out)                       :: status                  ! status flag

        real(p), allocatable :: hull_instrument(:,:), hull(:,:)
        integer, allocatable :: ihull(:)
        integer*8            :: ipointing

        call init_astrometry(header, status=status)
        if (status /= 0) return

        call convex_hull(coords, ihull)
        allocate (hull_instrument(2,size(ihull)), hull(2,size(ihull)))
        hull_instrument = coords(:,ihull)

        xmin = pInf
        xmax = mInf
        ymin = pInf
        ymax = mInf

#ifndef IFORT
        !$omp parallel do reduction(min:xmin,ymin) reduction(max:xmax,ymax) private(hull)
#endif
        do ipointing = 1, npointings

            call instrument2ad(hull_instrument, hull, size(ihull), ra(ipointing), dec(ipointing), pa(ipointing))
            hull = ad2xy_gnomonic(hull)
            xmin = min(xmin, minval(hull(1,:)))
            ymin = min(ymin, minval(hull(2,:)))
            xmax = max(xmax, maxval(hull(1,:)))
            ymax = max(ymax, maxval(hull(2,:)))

        end do
#ifndef IFORT
        !$omp end parallel do
#endif

    end subroutine instrument2xy_minmax


    !---------------------------------------------------------------------------


    subroutine instrument2pmatrix_nearest_neighbour(coords, ncoords, area, ra, dec, pa, masked, npointings, header, pmatrix, out,  &
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
                pmatrix(1,isample,:)%pixel = -1
                pmatrix(1,isample,:)%weight = 0
                cycle
            end if

            call instrument2ad(coords, coords2, int(ncoords), ra(isample), dec(isample), pa(isample))
            
            call ad2xys_gnomonic(coords2, x, y, s)

            ! the input map has flux densities, not surface brightness
            ! f_idetector = f_imap * weight
            ! with weight = detector_area / pixel_area
            ! and pixel_area = reference_area / s
            call xy2pmatrix(x, y, nx, ny, out, pmatrix(1,isample,:))
            pmatrix(1,isample,:)%weight = real(s * area, kind=kind(pmatrix%weight))

        end do
        !$omp end parallel do

    end subroutine instrument2pmatrix_nearest_neighbour


    !---------------------------------------------------------------------------


    subroutine instrument2pmatrix_sharp_edges(coords, ncoords, ra, dec, pa, masked, npointings, header, pmatrix,                   &
                                              npixels_per_sample, new_npixels_per_sample, out, status)
        !f2py integer*8, depend(npixels_per_sample,npointings,ncoords) :: pmatrix(npixels_per_sample*npointings*ncoords/4)
        integer*8, intent(in)                        :: ncoords, npointings ! #coordinates, #pointings
        real(p), intent(in)                          :: coords(2,ncoords)   ! instrument frame coordinates
        real(p), intent(in), dimension(npointings)   :: ra, dec, pa         ! input pointings in celestial coordinates
        logical*1, intent(in), dimension(npointings) :: masked              ! pointing flags: true if masked, removed
        character(len=*), intent(in)                 :: header              ! sky map FITS header
        type(PointingElement), intent(inout)         :: pmatrix(npixels_per_sample,npointings,ncoords/4) ! the pointing matrix
        integer, intent(in)  :: npixels_per_sample     ! input maximum number of sky pixels intersected by a detector
        integer, intent(out) :: new_npixels_per_sample ! actual maximum number of sky pixels intersected by a detector
        logical, intent(out) :: out                    ! true if some coordinates fall outside of the map
        integer, intent(out) :: status

        real(p)   :: coords2(2,ncoords)
        integer   :: roi(2,2,ncoords/4)
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
                pmatrix(:,isample,:)%pixel = -1
                pmatrix(:,isample,:)%weight = 0
                cycle
            end if

            call instrument2ad(coords, coords2, int(ncoords), ra(isample), dec(isample), pa(isample))
            
            coords2 = ad2xy_gnomonic(coords2)
            roi = xy2roi(coords2, 4)
            call roi2pmatrix(roi, 4, coords2, nx, ny, new_npixels_per_sample, out, pmatrix(:,isample,:))

        end do
        !$omp end parallel do

    end subroutine instrument2pmatrix_sharp_edges


end module wcsutils
