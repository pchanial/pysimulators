! Copyright 2010-2013 Pierre Chanial
! All rights reserved
!
module module_pointingmatrix

    use module_math_old,  only : NaN, nint_down, nint_up
    use module_geometry,  only : intersection_polygon_unity_square_r8
    use module_precision, only : sp
    use module_tamasis,   only : p
    implicit none
    private

    public :: PointingElement
    public :: PointingElement_i4_r4
    public :: PointingElement_i4_r8
    public :: PointingElement_i8_r4
    public :: PointingElement_i8_r8
    public :: PointingElementBlock_1_2_i4_r4
    public :: PointingElementBlock_1_2_i4_r8
    public :: PointingElementBlock_1_2_i8_r4
    public :: PointingElementBlock_1_2_i8_r8
    public :: PointingElementBlock_1_3_i4_r4
    public :: PointingElementBlock_1_3_i4_r8
    public :: PointingElementBlock_1_3_i8_r4
    public :: PointingElementBlock_1_3_i8_r8
    public :: PointingElementBlock_2_1_i4_r4
    public :: PointingElementBlock_2_1_i4_r8
    public :: PointingElementBlock_2_1_i8_r4
    public :: PointingElementBlock_2_1_i8_r8
    public :: PointingElementBlock_2_2_i4_r4
    public :: PointingElementBlock_2_2_i4_r8
    public :: PointingElementBlock_2_2_i8_r4
    public :: PointingElementBlock_2_2_i8_r8
    public :: PointingElementBlock_2_3_i4_r4
    public :: PointingElementBlock_2_3_i4_r8
    public :: PointingElementBlock_2_3_i8_r4
    public :: PointingElementBlock_2_3_i8_r8
    public :: PointingElementBlock_3_1_i4_r4
    public :: PointingElementBlock_3_1_i4_r8
    public :: PointingElementBlock_3_1_i8_r4
    public :: PointingElementBlock_3_1_i8_r8
    public :: PointingElementBlock_3_2_i4_r4
    public :: PointingElementBlock_3_2_i4_r8
    public :: PointingElementBlock_3_2_i8_r4
    public :: PointingElementBlock_3_2_i8_r8
    public :: PointingElementBlock_3_3_i4_r4
    public :: PointingElementBlock_3_3_i4_r8
    public :: PointingElementBlock_3_3_i8_r4
    public :: PointingElementBlock_3_3_i8_r8
    public :: PointingElementRot2d_i4_r4
    public :: PointingElementRot2d_i4_r8
    public :: PointingElementRot2d_i8_r4
    public :: PointingElementRot2d_i8_r8
    public :: PointingElementRot3d_i4_r4
    public :: PointingElementRot3d_i4_r8
    public :: PointingElementRot3d_i8_r4
    public :: PointingElementRot3d_i8_r8
    public :: pmatrix_direct
    public :: pmatrix_direct_one_pixel_per_sample
    public :: pmatrix_transpose
    public :: pmatrix_transpose_one_pixel_per_sample
    public :: pmatrix_ptp
    public :: pmatrix_mask
    public :: pmatrix_pack
    public :: xy2roi
    public :: xy2pmatrix
    public :: roi2pmatrix_cartesian
    public :: backprojection_weight
    public :: backprojection_weighted_roi

    type PointingElement
        integer*4 :: index
        real(sp)  :: value
    end type PointingElement

    type PointingElement_i4_r4
        integer*4 :: index
        real*4    :: value
    end type

    type PointingElement_i8_r4
        integer*8 :: index
        real*4    :: value
    end type

    type PointingElement_i4_r8
        integer*4 :: index
        real*8    :: value
    end type

    type PointingElement_i8_r8
        integer*8 :: index
        real*8    :: value
    end type

    type PointingElementBlock_1_2_i4_r4
        integer*4 :: index
        real*4    :: value(2,1)
    end type

    type PointingElementBlock_1_2_i8_r4
        integer*8 :: index
        real*4    :: value(2,1)
    end type

    type PointingElementBlock_1_2_i4_r8
        integer*4 :: index
        real*8    :: value(2,1)
    end type

    type PointingElementBlock_1_2_i8_r8
        integer*8 :: index
        real*8    :: value(2,1)
    end type

    type PointingElementBlock_1_3_i4_r4
        integer*4 :: index
        real*4    :: value(3,1)
    end type

    type PointingElementBlock_1_3_i8_r4
        integer*8 :: index
        real*4    :: value(3,1)
    end type

    type PointingElementBlock_1_3_i4_r8
        integer*4 :: index
        real*8    :: value(3,1)
    end type

    type PointingElementBlock_1_3_i8_r8
        integer*8 :: index
        real*8    :: value(3,1)
    end type

    type PointingElementBlock_2_1_i4_r4
        integer*4 :: index
        real*4    :: value(1,2)
    end type

    type PointingElementBlock_2_1_i8_r4
        integer*8 :: index
        real*4    :: value(1,2)
    end type

    type PointingElementBlock_2_1_i4_r8
        integer*4 :: index
        real*8    :: value(1,2)
    end type

    type PointingElementBlock_2_1_i8_r8
        integer*8 :: index
        real*8    :: value(1,2)
    end type

    type PointingElementBlock_2_2_i4_r4
        integer*4 :: index
        real*4    :: value(2,2)
    end type

    type PointingElementBlock_2_2_i8_r4
        integer*8 :: index
        real*4    :: value(2,2)
    end type

    type PointingElementBlock_2_2_i4_r8
        integer*4 :: index
        real*8    :: value(2,2)
    end type

    type PointingElementBlock_2_2_i8_r8
        integer*8 :: index
        real*8    :: value(2,2)
    end type

    type PointingElementBlock_2_3_i4_r4
        integer*4 :: index
        real*4    :: value(3,2)
    end type

    type PointingElementBlock_2_3_i8_r4
        integer*8 :: index
        real*4    :: value(3,2)
    end type

    type PointingElementBlock_2_3_i4_r8
        integer*4 :: index
        real*8    :: value(3,2)
    end type

    type PointingElementBlock_2_3_i8_r8
        integer*8 :: index
        real*8    :: value(3,2)
    end type

    type PointingElementBlock_3_1_i4_r4
        integer*4 :: index
        real*4    :: value(1,3)
    end type

    type PointingElementBlock_3_1_i8_r4
        integer*8 :: index
        real*4    :: value(1,3)
    end type

    type PointingElementBlock_3_1_i4_r8
        integer*4 :: index
        real*8    :: value(1,3)
    end type

    type PointingElementBlock_3_1_i8_r8
        integer*8 :: index
        real*8    :: value(1,3)
    end type

    type PointingElementBlock_3_2_i4_r4
        integer*4 :: index
        real*4    :: value(2,3)
    end type

    type PointingElementBlock_3_2_i8_r4
        integer*8 :: index
        real*4    :: value(2,3)
    end type

    type PointingElementBlock_3_2_i4_r8
        integer*4 :: index
        real*8    :: value(2,3)
    end type

    type PointingElementBlock_3_2_i8_r8
        integer*8 :: index
        real*8    :: value(2,3)
    end type

    type PointingElementBlock_3_3_i4_r4
        integer*4 :: index
        real*4    :: value(3,3)
    end type

    type PointingElementBlock_3_3_i8_r4
        integer*8 :: index
        real*4    :: value(3,3)
    end type

    type PointingElementBlock_3_3_i4_r8
        integer*4 :: index
        real*8    :: value(3,3)
    end type

    type PointingElementBlock_3_3_i8_r8
        integer*8 :: index
        real*8    :: value(3,3)
    end type

    type PointingElementRot2d_i4_r4
        integer*4 :: index
        real*4    :: r11, r21
    end type

    type PointingElementRot2d_i8_r4
        integer*8 :: index
        real*4    :: r11, r21
    end type

    type PointingElementRot2d_i4_r8
        integer*4 :: index
        real*8    :: r11, r21
    end type

    type PointingElementRot2d_i8_r8
        integer*8 :: index
        real*8    :: r11, r21
    end type

    type PointingElementRot3d_i4_r4
        integer*4 :: index
        real*4    :: r11, r22, r32
    end type

    type PointingElementRot3d_i8_r4
        integer*8 :: index
        real*4    :: r11, r22, r32
    end type

    type PointingElementRot3d_i4_r8
        integer*4 :: index
        real*8    :: r11, r22, r32
    end type

    type PointingElementRot3d_i8_r8
        integer*8 :: index
        real*8    :: r11, r22, r32
    end type


contains


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


    !-------------------------------------------------------------------------------------------------------------------------------


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


    !-------------------------------------------------------------------------------------------------------------------------------


    subroutine pmatrix_transpose(pmatrix, timeline, map)
        type(pointingelement), intent(in) :: pmatrix(:,:)
        real(kind=p), intent(in)          :: timeline(:)
        real(kind=p), intent(inout)       :: map(0:)
        integer                           :: isample, ipixel, npixels, nsamples

        npixels  = size(pmatrix, 1)
        nsamples = size(pmatrix, 2)

        !$omp parallel do
        do isample = 1, nsamples
            do ipixel = 1, npixels
                if (pmatrix(ipixel,isample)%index == -1) exit
                !$omp atomic
                map(pmatrix(ipixel,isample)%index) = map(pmatrix(ipixel,isample)%index) +                                          &
                    pmatrix(ipixel,isample)%value * timeline(isample)
            end do
        end do
        !$omp end parallel do

    end subroutine pmatrix_transpose


    !-------------------------------------------------------------------------------------------------------------------------------


    subroutine pmatrix_transpose_one_pixel_per_sample(pmatrix, timeline, map)
        type(pointingelement), intent(in) :: pmatrix(:)
        real(kind=p), intent(in)          :: timeline(:)
        real(kind=p), intent(inout)       :: map(0:)
        integer                           :: isample, nsamples

        nsamples = size(pmatrix, 1)

        !$omp parallel do
        do isample = 1, nsamples
            if (pmatrix(isample)%index >= 0) then
                !$omp atomic
                map(pmatrix(isample)%index) = map(pmatrix(isample)%index) + pmatrix(isample)%value * timeline(isample)
            end if
        end do
        !$omp end parallel do

    end subroutine pmatrix_transpose_one_pixel_per_sample


    !-------------------------------------------------------------------------------------------------------------------------------
   
   
    subroutine pmatrix_ptp(pmatrix, ptp)
        type(pointingelement), intent(in) :: pmatrix(:,:)
        real(kind=p), intent(inout)       :: ptp(0:,0:)
        integer                           :: isample
        integer                           :: ipixel, jpixel, i, j
        integer                           :: npixels, nsamples
        real(kind(pmatrix%value))         :: pi, pj
       
        npixels  = size(pmatrix, 1)
        nsamples = size(pmatrix, 2)
       
        !$omp parallel do private(isample, ipixel, jpixel, i, j, pi, pj)
        do isample = 1, nsamples
            do ipixel = 1, npixels
                if (pmatrix(ipixel,isample)%index == -1) exit
                i  = pmatrix(ipixel,isample)%index
                pi = pmatrix(ipixel,isample)%value
                do jpixel = 1, npixels
                    if (pmatrix(jpixel,isample)%index == -1) exit
                    j  = pmatrix(jpixel,isample)%index
                    pj = pmatrix(jpixel,isample)%value
                    !$omp atomic
                    ptp(i,j) = ptp(i,j) + pi * pj
                end do
            end do
        end do
        !$omp end parallel do

    end subroutine pmatrix_ptp


    !-------------------------------------------------------------------------------------------------------------------------------


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


    !-------------------------------------------------------------------------------------------------------------------------------


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


    !-------------------------------------------------------------------------------------------------------------------------------


    subroutine backprojection_weight(pmatrix, timeline, mask, map, weight)
        type(PointingElement), intent(in)     :: pmatrix(:,:)
        real(kind=p), intent(in)              :: timeline(:)
        logical(kind=1), intent(in), optional :: mask(:)
        real(kind=p), intent(inout)           :: map(0:)
        real(kind=p), intent(inout)           :: weight(0:)

        integer                               :: npixels, nsamples
        integer                               :: ipixel, isample, imap          
        logical                               :: domask

        npixels  = size(pmatrix, 1)
        nsamples = size(pmatrix, 2)
        domask   = present(mask)

        !$omp parallel do private(isample,ipixel,imap)
        do isample = 1, nsamples
            if (domask) then
                if (mask(isample)) cycle
            end if
            do ipixel = 1, npixels
                imap = pmatrix(ipixel,isample)%index
                if (imap == -1) exit
                !$omp atomic
                map(imap) = map(imap) + pmatrix(ipixel,isample)%value * timeline(isample)
                !$omp atomic
                weight(imap) = weight(imap) + pmatrix(ipixel,isample)%value
            end do
        end do
        !$omp end parallel do

    end subroutine backprojection_weight


    !-------------------------------------------------------------------------------------------------------------------------------


    ! backproject a frame into a minimap
    subroutine backprojection_weighted_roi(pmatrix, nx, timeline, mask, itime, map, roi)

        type(pointingelement), intent(in) :: pmatrix(:,:,:)
        integer, intent(in)               :: nx
        real(p), intent(in)               :: timeline(:,:)
        logical(kind=1), intent(in)       :: mask(:,:)
        integer, intent(in)               :: itime
        real(p), intent(out)              :: map(0:)
        integer, intent(out)              :: roi(2,2)

        real(sp) :: weight(0:ubound(map,1))
        integer  :: nxmap, xmap, ymap, imap
        integer  :: ndetectors, npixels_per_sample, idetector, ipixel

        npixels_per_sample = size(pmatrix,1)
        ndetectors = size(pmatrix,3)

        roi(1,1) = minval(modulo(pmatrix(:,itime,:)%index,nx), pmatrix(:,itime,:)%index /= -1) ! xmin
        roi(1,2) = maxval(modulo(pmatrix(:,itime,:)%index,nx), pmatrix(:,itime,:)%index /= -1) ! xmax
        roi(2,1) = minval(pmatrix(:,itime,:)%index / nx, pmatrix(:,itime,:)%index /= -1)       ! ymin
        roi(2,2) = maxval(pmatrix(:,itime,:)%index / nx, pmatrix(:,itime,:)%index /= -1)       ! ymax

        nxmap = roi(1,2) - roi(1,1) + 1

        ! backprojection of the timeline and weights
        map = 0
        weight = 0
        do idetector = 1, ndetectors

            if (mask(itime,idetector)) cycle

            do ipixel = 1, npixels_per_sample

                if (pmatrix(ipixel,itime,idetector)%index == -1) exit

                xmap = mod(pmatrix(ipixel,itime,idetector)%index, nx) - roi(1,1)
                ymap = pmatrix(ipixel,itime,idetector)%index / nx     - roi(2,1)
                imap = xmap + ymap * nxmap
                map(imap) = map(imap) + timeline(itime,idetector) * pmatrix(ipixel,itime,idetector)%value
                weight(imap) = weight(imap) + pmatrix(ipixel,itime,idetector)%value

            end do

        end do

        map = map / weight
        where (weight == 0)
            map = NaN
        end where

    end subroutine backprojection_weighted_roi


    !-------------------------------------------------------------------------------------------------------------------------------


    ! roi is a 3-dimensional array: [1=x|2=y,1=min|2=max,idetector]
    function xy2roi(xy) result(roi)

        real(p), intent(in) :: xy(:,:,:)

        integer             :: roi(size(xy,1),2,size(xy,3))
        integer             :: i

        do i = 1, size(xy,3)
            roi(1,1,i) = nint_up  (minval(xy(1,:,i)))
            roi(1,2,i) = nint_down(maxval(xy(1,:,i)))
            roi(2,1,i) = nint_up  (minval(xy(2,:,i)))
            roi(2,2,i) = nint_down(maxval(xy(2,:,i)))
        end do

    end function xy2roi


    !-------------------------------------------------------------------------------------------------------------------------------


    subroutine xy2pmatrix(x, y, nx, ny, out, pmatrix)

        real(p), intent(in)                :: x(:), y(:)
        integer, intent(in)                :: nx, ny
        logical, intent(inout)             :: out
        type(pointingelement), intent(out) :: pmatrix(:)

        integer                            :: idetector, ix, iy

        do idetector = 1, size(pmatrix)

            ix = nint_up(x(idetector))
            iy = nint_up(y(idetector))
            if (ix < 0 .or. ix > nx - 1 .or. iy < 0 .or. iy > ny - 1) then
               out = .true.
               pmatrix(idetector)%index  = -1
               pmatrix(idetector)%value = 0
               cycle
            end if

            pmatrix(idetector)%index  = ix + iy * nx
            pmatrix(idetector)%value = 1.

        end do

    end subroutine xy2pmatrix


    !-------------------------------------------------------------------------------------------------------------------------------


    subroutine roi2pmatrix_cartesian(roi, coords, nx, ny, new_npixels_per_sample, out, pmatrix)

        integer, intent(in)                :: roi(:,:,:)
        real(p), intent(in)                :: coords(:,:,:)
        integer, intent(in)                :: nx, ny
        integer, intent(inout)             :: new_npixels_per_sample
        logical, intent(inout)             :: out
        type(pointingelement), intent(out) :: pmatrix(:,:)

        integer  :: ncoords, nvertices
        real(p)  :: polygon(size(roi,1),size(coords,2))
        integer  :: npixels_per_sample, icoord, ix, iy, npixels
        real(kind(pmatrix%value)) :: val

        ncoords = size(roi, 3)
        nvertices = size(coords, 2)
        npixels_per_sample = size(pmatrix, 1)
        do icoord = 1, ncoords

            if (roi(2,1,icoord) < 0 .or. roi(2,2,icoord) > ny - 1 .or.   &
                roi(1,1,icoord) < 0 .or. roi(1,2,icoord) > nx - 1) then
               out = .true.
            end if

            npixels = 1
            do iy = max(roi(2,1,icoord), 0), min(roi(2,2,icoord), ny - 1)

                do ix = max(roi(1,1,icoord), 0), min(roi(1,2,icoord), nx - 1)

                    polygon(1,:) = coords(1,:,icoord) - (ix - 0.5_p)
                    polygon(2,:) = coords(2,:,icoord) - (iy - 0.5_p)
                    val = real(intersection_polygon_unity_square_r8(polygon, nvertices), sp)
                    if (val == 0) cycle
                    if (npixels <= npixels_per_sample) then
                        pmatrix(npixels,icoord)%index  = ix + iy * nx
                        pmatrix(npixels,icoord)%value = abs(val)
                    end if
                    npixels = npixels + 1

                end do

            end do

            ! fill the rest of the pointing matrix
            pmatrix(npixels:,icoord)%index  = -1
            pmatrix(npixels:,icoord)%value = 0
            new_npixels_per_sample = max(new_npixels_per_sample, npixels-1)

        end do

    end subroutine roi2pmatrix_cartesian


end module module_pointingmatrix
