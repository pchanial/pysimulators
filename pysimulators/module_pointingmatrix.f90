module pointingmatrix

    use module_precision, only : sp
    use module_tamasis, only : p

    implicit none

!    integer, parameter :: sp = selected_real_kind(6,37)  ! single precision
!    integer, parameter :: p = selected_real_kind(15,307) ! double precision

    type PointingElement
        real(sp) :: value
        integer  :: index
    end type PointingElement

contains
    
    subroutine backprojection_weight(pmatrix, data, map1d, weight1d, npixels_per_sample, nsamples, npixels)

        !f2py integer*8,intent(in)        :: pmatrix(npixels_per_sample*nsamples)

        integer*8, intent(in)             :: nsamples
        type(PointingElement), intent(in) :: pmatrix(npixels_per_sample, nsamples)
        real(p), intent(in)               :: data(nsamples)
        real(p), intent(inout)            :: map1d(npixels)
        real(p), intent(inout)            :: weight1d(npixels)
        integer, intent(in)               :: npixels_per_sample
        integer, intent(in)               :: npixels

        call backprojection_weight__inner(pmatrix, data, map=map1d, weight=weight1d)

    end subroutine backprojection_weight


    !---------------------------------------------------------------------------


    subroutine backprojection_weight_mask(pmatrix, data, mask, map1d, weight1d, npixels_per_sample, nsamples, npixels)

        !f2py integer*8,intent(in)        :: pmatrix(npixels_per_sample*nsamples)

        integer*8, intent(in)             :: nsamples
        type(PointingElement), intent(in) :: pmatrix(npixels_per_sample, nsamples)
        real(p), intent(in)               :: data(nsamples)
        logical*1, intent(in)             :: mask(nsamples)
        real(p), intent(inout)            :: map1d(npixels)
        real(p), intent(inout)            :: weight1d(npixels)
        integer, intent(in)               :: npixels_per_sample
        integer, intent(in)               :: npixels

        call backprojection_weight__inner(pmatrix, data, mask, map1d, weight1d)

    end subroutine backprojection_weight_mask


    !---------------------------------------------------------------------------


    subroutine check(matrix, npixels_per_sample, nsamples, npixels, isvalid)
        !f2py threadsafe
        !f2py integer*8, dimension(npixels_per_sample*nsamples) :: matrix

        integer*8, intent(in)             :: nsamples
        type(PointingElement), intent(in) :: matrix(npixels_per_sample,nsamples)
        integer, intent(in)               :: npixels_per_sample
        integer, intent(in)               :: npixels
        logical*1, intent(out)            :: isvalid
        integer :: index, ipixel, isample

        isvalid = .false.
        do isample = 1, size(matrix, 2)
            do ipixel = 1, size(matrix, 1)
                index = matrix(ipixel,isample)%index
                if (index == -1) cycle
                if (index < 0 .or. index >= npixels) return
            end do
        end do
        isvalid = .true.

    end subroutine check


    !---------------------------------------------------------------------------


    subroutine direct(pmatrix, map1d, signal, npixels_per_sample, nsamples, npixels)

        !f2py integer*8, dimension(npixels_per_sample*nsamples), intent(inout) :: pmatrix

        integer*8, intent(in)  :: nsamples
        type(PointingElement), intent(inout) :: pmatrix(npixels_per_sample, nsamples)
        real(p), intent(in)    :: map1d(npixels)
        real(p), intent(inout) :: signal(nsamples)
        integer, intent(in)    :: npixels_per_sample
        integer, intent(in)    :: npixels

        if (npixels_per_sample == 1) then
            call pmatrix_direct_one_pixel_per_sample(pmatrix(1,:), map1d, signal)
        else
            call pmatrix_direct(pmatrix, map1d, signal)
        end if

    end subroutine direct


    !---------------------------------------------------------------------------


    subroutine intersects(pmatrix, pixel, npixels_per_sample, nsamples, ndetectors, out)
        !f2py integer*8, dimension(npixels_per_sample*nsamples*ndetectors), intent(in) :: pmatrix

        integer*8, intent(in)                     :: nsamples
        type(PointingElement), intent(in)         :: pmatrix(npixels_per_sample, nsamples, ndetectors)
        integer(kind(pmatrix%value)), intent(in) :: pixel
        integer, intent(in)                       :: npixels_per_sample
        integer, intent(in)                       :: ndetectors
        logical*1, intent(out)                    :: out
        
        out = any(pmatrix%index == pixel)

    end subroutine intersects


    !---------------------------------------------------------------------------


    subroutine intersects_openmp2(pmatrix, pixel, npixels_per_sample, nsamples, ndetectors, out)
        !f2py integer*8, dimension(npixels_per_sample*nsamples*ndetectors), intent(in) :: pmatrix

        integer*8, intent(in)                     :: nsamples
        type(PointingElement), intent(in)         :: pmatrix(npixels_per_sample, nsamples, ndetectors)
        integer(kind(pmatrix%value)), intent(in) :: pixel
        integer, intent(in)                       :: npixels_per_sample
        integer, intent(in)                       :: ndetectors
        logical*1, intent(out)                    :: out
        
        !$omp parallel workshare
        out = any(pmatrix%index == pixel)
        !$omp end parallel workshare

    end subroutine intersects_openmp2


    !---------------------------------------------------------------------------


    subroutine intersects_axis2(pmatrix, pixel, npixels_per_sample, nsamples, ndetectors, out)
        !f2py integer*8, dimension(npixels_per_sample*nsamples*ndetectors), intent(in) :: pmatrix

        integer*8, intent(in)                     :: nsamples
        type(PointingElement), intent(in)         :: pmatrix(npixels_per_sample, nsamples, ndetectors)
        integer(kind(pmatrix%value)), intent(in) :: pixel
        integer, intent(in)                       :: npixels_per_sample
        integer, intent(in)                       :: ndetectors
        logical*1, intent(out)                    :: out(ndetectors)

        integer   :: idetector, ipixel
        integer*8 :: isample

        out = .false.
        !$omp parallel do schedule(guided)
        loop_detector: do idetector = 1, ndetectors
            do isample = 1, nsamples
                do ipixel = 1, npixels_per_sample
                    if (pmatrix(ipixel,isample,idetector)%index == pixel) then
                        out(idetector) = .true.
                        cycle loop_detector
                    end if
                end do
            end do
        end do loop_detector
        !$omp end parallel do

    end subroutine intersects_axis2


    !---------------------------------------------------------------------------


    subroutine intersects_axis3(pmatrix, pixel, npixels_per_sample, nsamples, ndetectors, out)
        !f2py integer*8, dimension(npixels_per_sample*nsamples*ndetectors), intent(in) :: pmatrix

        integer*8, intent(in)                     :: nsamples
        type(PointingElement), intent(in)         :: pmatrix(npixels_per_sample, nsamples, ndetectors)
        integer(kind(pmatrix%value)), intent(in) :: pixel
        integer, intent(in)                       :: npixels_per_sample
        integer, intent(in)                       :: ndetectors
        logical*1, intent(out)                    :: out(nsamples)

        integer   :: idetector, ipixel
        integer*8 :: isample

        out = .false.
        !$omp parallel do schedule(guided)
        loop_sample: do isample = 1, nsamples
            do idetector = 1, ndetectors
                do ipixel = 1, npixels_per_sample
                    if (pmatrix(ipixel,isample,idetector)%index == pixel) then
                        out(isample) = .true.
                        cycle loop_sample
                    end if
                end do
            end do
        end do loop_sample
        !$omp end parallel do

    end subroutine intersects_axis3


    !---------------------------------------------------------------------------


    subroutine mask(pmatrix, mask1d, npixels_per_sample, nsamples, npixels)

        !f2py integer*8, dimension(npixels_per_sample*nsamples), intent(in) :: pmatrix

        integer*8, intent(in)             :: nsamples
        type(PointingElement), intent(in) :: pmatrix(npixels_per_sample, nsamples)
        logical*1, intent(inout)          :: mask1d(npixels)
        integer, intent(in)               :: npixels_per_sample
        integer, intent(in)               :: npixels

        call pmatrix_mask(pmatrix, mask1d)

    end subroutine mask


    !---------------------------------------------------------------------------
    
    
    subroutine pack(pmatrix, mask1d, npixels_per_sample, nsamples, npixels)

        !f2py integer*8, dimension(npixels_per_sample*nsamples), intent(inout) :: pmatrix

        integer*8, intent(in)                :: nsamples
        type(PointingElement), intent(inout) :: pmatrix(npixels_per_sample, nsamples)
        logical*1, intent(in)                :: mask1d(npixels)
        integer, intent(in)                  :: npixels_per_sample
        integer, intent(in)                  :: npixels

        call pmatrix_pack(pmatrix, mask1d)

    end subroutine pack


    !---------------------------------------------------------------------------


    subroutine ptp(pmatrix, array, npixels_per_sample, nsamples, npixels)

        !f2py integer*8, dimension(npixels_per_sample*nsamples), intent(inout) :: pmatrix

        integer*8, intent(in)                :: nsamples
        type(PointingElement), intent(inout) :: pmatrix(npixels_per_sample, nsamples)
        real(p), intent(inout)               :: array(npixels, npixels)
        integer, intent(in)                  :: npixels_per_sample
        integer, intent(in)                  :: npixels

        call pmatrix_ptp(pmatrix, array)

    end subroutine ptp


    !---------------------------------------------------------------------------


    subroutine transpose(pmatrix, signal, map1d, npixels_per_sample, nsamples, npixels)

        !f2py integer*8, dimension(npixels_per_sample*nsamples), intent(inout) :: pmatrix

        integer*8, intent(in)  :: nsamples
        type(PointingElement), intent(inout) :: pmatrix(npixels_per_sample, nsamples)
        real(p), intent(in)    :: signal(nsamples)
        real(p), intent(inout) :: map1d(npixels)
        integer, intent(in)    :: npixels_per_sample
        integer, intent(in)    :: npixels

        if (npixels_per_sample == 1) then
            call pmatrix_transpose_one_pixel_per_sample(pmatrix(1,:), signal, map1d)
        else
            call pmatrix_transpose(pmatrix, signal, map1d)
        end if

    end subroutine transpose


    !---------------------------------------------------------------------------


    subroutine roi2matrix_cartesian(roi, coords, ncoords, nvertices,           &
        npixels_per_sample, nx, ny, matrix, new_npixels_per_sample, out)
        !f2py threadsafe
        !f2py integer*8, dimension(npixels_per_sample*ncoords) :: matrix

        integer, intent(in)                 :: roi(2,2,ncoords)
        real(p), intent(in)                 :: coords(2,nvertices,ncoords)
        integer, intent(in)                 :: ncoords, nvertices
        integer, intent(in)                 :: npixels_per_sample
        integer, intent(in)                 :: nx, ny
        integer, intent(out)                :: new_npixels_per_sample
        logical, intent(out)                :: out
        type(PointingElement), intent(inout)::matrix(npixels_per_sample,ncoords)

        real(p) :: polygon(2,nvertices)
        integer :: icoord, ix, iy, npixels
        real*4  :: val

        do icoord = 1, size(matrix, 2)

            if (roi(2,1,icoord) < 1 .or. roi(2,2,icoord) > ny .or.             &
                roi(1,1,icoord) < 1 .or. roi(1,2,icoord) > nx) then
               out = .true.
            end if

            npixels = 1
            do iy = max(roi(2,1,icoord),1), min(roi(2,2,icoord),ny)

                do ix = max(roi(1,1,icoord),1), min(roi(1,2,icoord),nx)

                    polygon(1,:) = coords(1,:,icoord) - (ix-0.5_p)
                    polygon(2,:) = coords(2,:,icoord) - (iy-0.5_p)
                    val = real(abs(intersection_polygon_unity_square(polygon, nvertices)), kind=sp)
                    if (val == 0) cycle
                    if (npixels <= npixels_per_sample) then
                        matrix(npixels,icoord)%index = ix - 1 + (iy - 1) * nx
                        matrix(npixels,icoord)%value = val
                    end if
                    npixels = npixels + 1

                end do

            end do

            ! fill the rest of the pointing matrix
            matrix(npixels:,icoord)%index  = -1
            matrix(npixels:,icoord)%value = 0
            new_npixels_per_sample = max(new_npixels_per_sample, npixels-1)

        end do

    end subroutine roi2matrix_cartesian


    !---------------------------------------------------------------------------
    !---------------------------------------------------------------------------
    !---------------------------------------------------------------------------
    !---------------------------------------------------------------------------


    pure function intersection_polygon_unity_square(xy, nvertices) result(out)

        real*8              :: out
        real*8, intent(in)  :: xy(2,nvertices)
        integer, intent(in) :: nvertices
        integer             :: i, j

        out = 0
        j = nvertices
        do i=1, nvertices
            out = out + intersection_segment_unity_square(xy(1,i), xy(2,i),    &
                                                          xy(1,j), xy(2,j))
            j = i
        end do

    end function intersection_polygon_unity_square


    !---------------------------------------------------------------------------


    pure function intersection_segment_unity_square(x1, y1, x2, y2) result(out)

        real*8             :: out
        real*8, intent(in) :: x1, y1, x2, y2 ! 1st and 2nd point coordinates

        real*8 :: pente       ! slope of the straight line going through p1, p2
        real*8 :: ordonnee    ! point where the straight line crosses y-axis
        real*8 :: delta_x     ! = x2-x1
        real*8 :: xmin, xmax  ! minimum and maximum values of x to consider
                              ! (clipped in the square (0,0),(1,0),(1,1),(0,1)
        real*8 :: ymin, ymax  ! minimum and maximum values of y to consider
                              ! (clipped in the square (0,0),(1,0),(1,1),(0,1)
        real*8 :: xhaut       ! value of x at which straight line crosses the
                              ! (0,1),(1,1) line
        logical:: neg_delta_x ! TRUE if delta_x < 0

        ! Check for vertical line : the area intercepted is 0
        if (x1 == x2) then
            out = 0
            return
        end if

        ! Order the two input points in x
        if (x2 > x1) then
            xmin = x1
            xmax = x2
        else
            xmin = x2
            xmax = x1
        end if

        ! And determine the bounds ignoring y for now
        ! test is p1 and p2 are outside the square along x-axis
        if (xmin > 1 .or. xmax < 0) then
            out = 0
            return    ! outside, the area is 0
        end if

        ! We compute xmin, xmax, clipped between 0 and 1 in x
        ! then we compute pente (slope) and ordonnee and use it to get ymin
        ! and ymax
        xmin = max(xmin, 0._p)
        xmax = min(xmax, 1._p)

        delta_x = x2 - x1
        neg_delta_x = delta_x < 0
        pente = (y2 - y1) / delta_x
        ordonnee  = y1 - pente * x1
        ymin = pente * xmin + ordonnee
        ymax = pente * xmax + ordonnee

        ! Trap segment entirely below axis
        if (ymin < 0 .and. ymax < 0) then
            out = 0
            return  ! if segment below axis, intercepted surface is 0
        end if

        ! Adjust bounds if segment crosses axis x-axis
        !(to exclude anything below axis)
        if (ymin < 0) then
            ymin = 0
            xmin = - ordonnee / pente
        end if
        if (ymax < 0) then
            ymax = 0
            xmax = - ordonnee / pente
        end if

        ! There are four possibilities: both y below 1, both y above 1
        ! and one of each.

        if (ymin >= 1 .and. ymax >= 1) then

            ! Line segment is entirely above square : we clip with the square
            if (neg_delta_x) then
                out = xmin - xmax
            else
                out = xmax - xmin
            end if
            return

        end if

        if (ymin <= 1 .and. ymax <= 1) then
          ! Segment is entirely within square
          if (neg_delta_x) then
             out = 0.5_p * (xmin-xmax) * (ymax+ymin)
          else
             out = 0.5_p * (xmax-xmin) * (ymax+ymin)
          end if
          return
        end if

        ! otherwise it must cross the top of the square
        ! the crossing occurs at xhaut
        xhaut = (1 - ordonnee) / pente
        !!if ((xhaut < xmin) .or. (xhaut > xmax))   cout << " BUGGGG "
        if (ymin < 1) then
            if (neg_delta_x) then
                out = -(0.5_p * (xhaut-xmin) * (1+ymin) + xmax - xhaut)
            else
                out = 0.5_p * (xhaut-xmin) * (1+ymin) + xmax - xhaut
            end if
        else
            if (neg_delta_x) then
                out = -(0.5_p * (xmax-xhaut) * (1+ymax) + xhaut-xmin)
            else
                out = 0.5_p * (xmax-xhaut) * (1+ymax) + xhaut-xmin
            end if
        end if

    end function intersection_segment_unity_square


    subroutine pmatrix_direct(pmatrix, map, timeline)

        type(pointingelement), intent(in) :: pmatrix(:,:)
        real(p), intent(in)               :: map(0:)
        real(p), intent(inout)            :: timeline(:)
        integer                           :: ipixel, isample, npixels_per_sample, nsamples

        npixels_per_sample = size(pmatrix,1)
        nsamples = size(pmatrix, 2)

        !$omp parallel do
        do isample = 1, nsamples
!!$            timeline(isample) = sum(map(pmatrix(:,isample)%index) * pmatrix(:,isample)%value)
            timeline(isample) = 0
            do ipixel = 1, npixels_per_sample
                if (pmatrix(ipixel,isample)%index == -1) exit
                timeline(isample) = timeline(isample) + map(pmatrix(ipixel,isample)%index) * pmatrix(ipixel,isample)%value
            end do
        end do
        !$omp end parallel do

    end subroutine pmatrix_direct


    !---------------------------------------------------------------------------


    subroutine pmatrix_direct_one_pixel_per_sample(pmatrix, map, timeline)

        type(pointingelement), intent(in) :: pmatrix(:)
        real(p), intent(in)               :: map(0:)
        real(p), intent(inout)            :: timeline(:)
        integer                           :: isample, nsamples

        nsamples = size(pmatrix, 1)

        !$omp parallel do
        do isample = 1, nsamples
            if (pmatrix(isample)%index >= 0) then
                timeline(isample) = map(pmatrix(isample)%index) * pmatrix(isample)%value
            end if
        end do
        !$omp end parallel do

    end subroutine pmatrix_direct_one_pixel_per_sample


    !---------------------------------------------------------------------------


    subroutine pmatrix_transpose(pmatrix, timeline, map)
        type(pointingelement), intent(in) :: pmatrix(:,:)
        real(p), intent(in)          :: timeline(:)
        real(p), intent(out)         :: map(0:)
        integer                           :: isample, ipixel, npixels, nsamples

        npixels  = size(pmatrix, 1)
        nsamples = size(pmatrix, 2)

#ifdef GFORTRAN
        !$omp parallel do reduction(+:map)
#else
        !$omp parallel do
#endif
        do isample = 1, nsamples
            do ipixel = 1, npixels
                if (pmatrix(ipixel,isample)%index == -1) exit
#ifndef GFORTRAN
                !$omp atomic
#endif
                map(pmatrix(ipixel,isample)%index) = map(pmatrix(ipixel,isample)%index) +                                          &
                    pmatrix(ipixel,isample)%value * timeline(isample)
            end do
        end do
        !$omp end parallel do

    end subroutine pmatrix_transpose


    !---------------------------------------------------------------------------


    subroutine pmatrix_transpose_one_pixel_per_sample(pmatrix, timeline, map)
        type(pointingelement), intent(in) :: pmatrix(:)
        real(p), intent(in)          :: timeline(:)
        real(p), intent(out)         :: map(0:)
        integer                           :: isample, nsamples

        nsamples = size(pmatrix, 1)

        do isample = 1, nsamples
            if (pmatrix(isample)%index >= 0) then
                map(pmatrix(isample)%index) = map(pmatrix(isample)%index) + pmatrix(isample)%value * timeline(isample)
            end if
        end do

    end subroutine pmatrix_transpose_one_pixel_per_sample


    !---------------------------------------------------------------------------
   
   
    subroutine pmatrix_ptp(pmatrix, ptp)
        type(pointingelement), intent(in) :: pmatrix(:,:)
        real(p), intent(inout)       :: ptp(0:,0:)
        integer                           :: isample
        integer                           :: ipixel, jpixel, i, j
        integer                           :: npixels, nsamples
        real(kind(pmatrix%value))        :: pi, pj
       
        npixels  = size(pmatrix, 1)
        nsamples = size(pmatrix, 2)
       
        !$omp parallel do reduction(+:ptp) private(isample, ipixel, jpixel, i, j, pi, pj)
        do isample = 1, nsamples
            do ipixel = 1, npixels
                if (pmatrix(ipixel,isample)%index == -1) exit
                i  = pmatrix(ipixel,isample)%index
                pi = pmatrix(ipixel,isample)%value
                do jpixel = 1, npixels
                    if (pmatrix(jpixel,isample)%index == -1) exit
                    j  = pmatrix(jpixel,isample)%index
                    pj = pmatrix(jpixel,isample)%value
                    ptp(i,j) = ptp(i,j) + pi * pj
                end do
            end do
        end do
        !$omp end parallel do

    end subroutine pmatrix_ptp


    !---------------------------------------------------------------------------


    subroutine pmatrix_mask(pmatrix, mask)
        ! True means: not observed
        type(pointingelement), intent(in) :: pmatrix(:,:)
        logical(1), intent(inout)         :: mask(0:)

        integer*8 :: isample, ipixel, npixels, nsamples
        integer*4 :: pixel

        npixels  = size(pmatrix, 1)
        nsamples = size(pmatrix, 2)

        !$omp parallel do private(pixel)
        do isample = 1, nsamples
            do ipixel = 1, npixels
                pixel = pmatrix(ipixel,isample)%index
                if (pixel == -1) exit
                if (pmatrix(ipixel,isample)%value == 0) cycle
                mask(pixel) = .false.
            end do
        end do
        !$omp end parallel do

    end subroutine pmatrix_mask


    !---------------------------------------------------------------------------


    subroutine pmatrix_pack(pmatrix, mask)
        ! True means: not observed

        type(pointingelement), intent(inout) :: pmatrix(:,:)
        logical(1), intent(in)               :: mask(0:)

        integer*8 :: isample, nsamples
        integer*4 :: table(lbound(mask,1):ubound(mask,1))
        integer*4 :: pixel, ipixel, ipacked, npixels

        ! fill a table which contains the packed indices of the non-masked pixels
        ipacked = 0
        do ipixel=0, size(mask)-1
            if (mask(ipixel)) cycle
            table(ipixel) = ipacked
            ipacked = ipacked + 1
        end do

        npixels  = size(pmatrix, 1)
        nsamples = size(pmatrix, 2)

        !$omp parallel do private(pixel, ipixel)
        do isample = 1, nsamples
            ipixel = 1
            do while (ipixel <= npixels)
                pixel = pmatrix(ipixel,isample)%index
                if (pixel == -1) exit
                if (pmatrix(ipixel,isample)%value == 0 .or. mask(pixel)) then
                    pmatrix(ipixel:npixels-1,isample) = pmatrix(ipixel+1:npixels,isample)
                    pmatrix(npixels,isample)%index = -1
                    pmatrix(npixels,isample)%value = 0
                    cycle
                end if
                pmatrix(ipixel,isample)%index = table(pixel)
                ipixel = ipixel + 1
            end do
        end do
        !$omp end parallel do

    end subroutine pmatrix_pack


    !---------------------------------------------------------------------------


    subroutine backprojection_weight__inner(pmatrix, timeline, mask, map, weight)
        type(pointingelement), intent(in)     :: pmatrix(:,:)
        real(p), intent(in)              :: timeline(:)
        logical(kind=1), intent(in), optional :: mask(:)
        real(p), intent(inout)           :: map(0:)
        real(p), intent(inout)           :: weight(0:)

        integer                               :: npixels, nsamples
        integer                               :: ipixel, isample, imap          
        logical                               :: domask

        npixels  = size(pmatrix, 1)
        nsamples = size(pmatrix, 2)
        domask   = present(mask)

        !$omp parallel do &
#ifdef GFORTRAN
        !$omp reduction(+:map,weight) &
#endif
        !$omp private(isample,ipixel,imap)
        do isample = 1, nsamples
            if (domask) then
                if (mask(isample)) cycle
            end if
            do ipixel = 1, npixels
                imap = pmatrix(ipixel,isample)%index
                if (imap == -1) exit
#ifndef GFORTRAN
                !$omp atomic
#endif
                map   (imap) = map   (imap) + pmatrix(ipixel,isample)%value * timeline(isample)
#ifndef GFORTRAN
                !$omp atomic
#endif
                weight(imap) = weight(imap) + pmatrix(ipixel,isample)%value
            end do
        end do
        !$omp end parallel do

    end subroutine backprojection_weight__inner


end module pointingmatrix
