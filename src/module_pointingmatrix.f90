! Copyright 2010-2011 Pierre Chanial
! All rights reserved
!
module module_pointingmatrix

    use module_math,       only : NaN, nint_down, nint_up
    use module_projection, only : intersection_polygon_unity_square
    use module_precision,  only : sp
    use module_tamasis,    only : p
    implicit none
    private

    public :: pointingelement
    public :: pmatrix_direct
    public :: pmatrix_direct_one_pixel_per_sample
    public :: pmatrix_transpose
    public :: pmatrix_transpose_one_pixel_per_sample
    public :: pmatrix_ptp
    public :: pmatrix_mask
    public :: pmatrix_pack
    public :: xy2roi
    public :: xy2pmatrix
    public :: roi2pmatrix
    public :: backprojection_weight
    public :: backprojection_weighted_roi

    type pointingelement
        real(sp)  :: weight
        integer*4 :: pixel
    end type pointingelement

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
!!$            timeline(isample) = sum(map(pmatrix(:,isample)%pixel) * pmatrix(:,isample)%weight)
            timeline(isample) = 0
            do ipixel = 1, npixels_per_sample
                if (pmatrix(ipixel,isample)%pixel == -1) exit
                timeline(isample) = timeline(isample) + map(pmatrix(ipixel,isample)%pixel) * pmatrix(ipixel,isample)%weight
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
            if (pmatrix(isample)%pixel >= 0) then
                timeline(isample) = map(pmatrix(isample)%pixel) * pmatrix(isample)%weight
            end if
        end do
        !$omp end parallel do

    end subroutine pmatrix_direct_one_pixel_per_sample


    !-------------------------------------------------------------------------------------------------------------------------------


    subroutine pmatrix_transpose(pmatrix, timeline, map)
        type(pointingelement), intent(in) :: pmatrix(:,:)
        real(kind=p), intent(in)          :: timeline(:)
        real(kind=p), intent(out)         :: map(0:)
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
                if (pmatrix(ipixel,isample)%pixel == -1) exit
#ifndef GFORTRAN
                !$omp atomic
#endif
                map(pmatrix(ipixel,isample)%pixel) = map(pmatrix(ipixel,isample)%pixel) +                                          &
                    pmatrix(ipixel,isample)%weight * timeline(isample)
            end do
        end do
        !$omp end parallel do

    end subroutine pmatrix_transpose


    !-------------------------------------------------------------------------------------------------------------------------------


    subroutine pmatrix_transpose_one_pixel_per_sample(pmatrix, timeline, map)
        type(pointingelement), intent(in) :: pmatrix(:)
        real(kind=p), intent(in)          :: timeline(:)
        real(kind=p), intent(out)         :: map(0:)
        integer                           :: isample, nsamples

        nsamples = size(pmatrix, 1)

        do isample = 1, nsamples
            if (pmatrix(isample)%pixel >= 0) then
                map(pmatrix(isample)%pixel) = map(pmatrix(isample)%pixel) + pmatrix(isample)%weight * timeline(isample)
            end if
        end do

    end subroutine pmatrix_transpose_one_pixel_per_sample


    !-------------------------------------------------------------------------------------------------------------------------------
   
   
    subroutine pmatrix_ptp(pmatrix, ptp)
        type(pointingelement), intent(in) :: pmatrix(:,:)
        real(kind=p), intent(inout)       :: ptp(0:,0:)
        integer                           :: isample
        integer                           :: ipixel, jpixel, i, j
        integer                           :: npixels, nsamples
        real(kind(pmatrix%weight))        :: pi, pj
       
        npixels  = size(pmatrix, 1)
        nsamples = size(pmatrix, 2)
       
        !$omp parallel do reduction(+:ptp) private(isample, ipixel, jpixel, i, j, pi, pj)
        do isample = 1, nsamples
            do ipixel = 1, npixels
                if (pmatrix(ipixel,isample)%pixel == -1) exit
                i  = pmatrix(ipixel,isample)%pixel
                pi = pmatrix(ipixel,isample)%weight
                do jpixel = 1, npixels
                    if (pmatrix(jpixel,isample)%pixel == -1) exit
                    j  = pmatrix(jpixel,isample)%pixel
                    pj = pmatrix(jpixel,isample)%weight
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
                pixel = pmatrix(ipixel,isample)%pixel
                if (pixel == -1) exit
                if (pmatrix(ipixel,isample)%weight == 0) cycle
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
                pixel = pmatrix(ipixel,isample)%pixel
                if (pixel == -1) exit
                if (pmatrix(ipixel,isample)%weight == 0 .or. mask(pixel)) then
                    pmatrix(ipixel:npixels-1,isample) = pmatrix(ipixel+1:npixels,isample)
                    pmatrix(npixels,isample)%pixel = -1
                    pmatrix(npixels,isample)%weight = 0
                    cycle
                end if
                pmatrix(ipixel,isample)%pixel = table(pixel)
                ipixel = ipixel + 1
            end do
        end do
        !$omp end parallel do

    end subroutine pmatrix_pack


    !-------------------------------------------------------------------------------------------------------------------------------


    subroutine backprojection_weight(pmatrix, timeline, mask, map, weight)
        type(pointingelement), intent(in)     :: pmatrix(:,:)
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
                imap = pmatrix(ipixel,isample)%pixel
                if (imap == -1) exit
#ifndef GFORTRAN
                !$omp atomic
#endif
                map   (imap) = map   (imap) + pmatrix(ipixel,isample)%weight * timeline(isample)
#ifndef GFORTRAN
                !$omp atomic
#endif
                weight(imap) = weight(imap) + pmatrix(ipixel,isample)%weight
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

        roi(1,1) = minval(modulo(pmatrix(:,itime,:)%pixel,nx), pmatrix(:,itime,:)%pixel /= -1) + 1 ! xmin
        roi(1,2) = maxval(modulo(pmatrix(:,itime,:)%pixel,nx), pmatrix(:,itime,:)%pixel /= -1) + 1 ! xmax
        roi(2,1) = minval(pmatrix(:,itime,:)%pixel / nx, pmatrix(:,itime,:)%pixel /= -1) + 1       ! ymin
        roi(2,2) = maxval(pmatrix(:,itime,:)%pixel / nx, pmatrix(:,itime,:)%pixel /= -1) + 1       ! ymax

        nxmap = roi(1,2) - roi(1,1) + 1

        ! backprojection of the timeline and weights
        map = 0
        weight = 0
        do idetector = 1, ndetectors

            if (mask(itime,idetector)) cycle

            do ipixel = 1, npixels_per_sample

                if (pmatrix(ipixel,itime,idetector)%pixel == -1) exit

                xmap = mod(pmatrix(ipixel,itime,idetector)%pixel, nx) - roi(1,1) + 1
                ymap = pmatrix(ipixel,itime,idetector)%pixel / nx     - roi(2,1) + 1
                imap = xmap + ymap * nxmap
                map(imap) = map(imap) + timeline(itime,idetector) * pmatrix(ipixel,itime,idetector)%weight
                weight(imap) = weight(imap) + pmatrix(ipixel,itime,idetector)%weight

            end do

        end do

        map = map / weight
        where (weight == 0)
            map = NaN
        end where

    end subroutine backprojection_weighted_roi


    !-------------------------------------------------------------------------------------------------------------------------------


    ! roi is a 3-dimensional array: [1=x|2=y,1=min|2=max,idetector]
    function xy2roi(xy, nvertices) result(roi)

        real(p), intent(in) :: xy(:,:)
        integer, intent(in) :: nvertices

        integer             :: roi(size(xy,1),2,size(xy,2)/nvertices)
        integer             :: idetector

        do idetector = 1, size(xy,2) / nvertices
            roi(1,1,idetector) = nint_up  (minval(xy(1,nvertices * (idetector-1)+1:nvertices*idetector)))
            roi(1,2,idetector) = nint_down(maxval(xy(1,nvertices * (idetector-1)+1:nvertices*idetector)))
            roi(2,1,idetector) = nint_up  (minval(xy(2,nvertices * (idetector-1)+1:nvertices*idetector)))
            roi(2,2,idetector) = nint_down(maxval(xy(2,nvertices * (idetector-1)+1:nvertices*idetector)))
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
            if (ix < 1 .or. ix > nx .or. iy < 1 .or. iy > ny) then
               out = .true.
               pmatrix(idetector)%pixel  = -1
               pmatrix(idetector)%weight = 0
               cycle
            end if

            pmatrix(idetector)%pixel  = ix - 1 + (iy - 1) * nx
            pmatrix(idetector)%weight = 1.

        end do

    end subroutine xy2pmatrix


    !-------------------------------------------------------------------------------------------------------------------------------


    subroutine roi2pmatrix(roi, nvertices, coords, nx, ny, nroi, out, pmatrix)

        integer, intent(in)                :: roi(:,:,:)
        integer, intent(in)                :: nvertices
        real(p), intent(in)                :: coords(:,:)
        integer, intent(in)                :: nx, ny
        integer, intent(inout)             :: nroi
        logical, intent(inout)             :: out
        type(pointingelement), intent(out) :: pmatrix(:,:)

        real(p)  :: polygon(size(roi,1),nvertices)
        integer  :: npixels_per_sample, idetector, ix, iy, iroi
        real(sp) :: weight

        npixels_per_sample = size(pmatrix, 1)
        do idetector = 1, size(pmatrix, 2)

            if (roi(2,1,idetector) < 1 .or. roi(2,2,idetector) > ny .or. roi(1,1,idetector) < 1 .or. roi(1,2,idetector) > nx) then
               out = .true.
            end if

            iroi = 1
            do iy = max(roi(2,1,idetector),1), min(roi(2,2,idetector),ny)

                do ix = max(roi(1,1,idetector),1), min(roi(1,2,idetector),nx)

                    polygon(1,:) = coords(1,(idetector-1)*nvertices+1:idetector*nvertices) - (ix-0.5_p)
                    polygon(2,:) = coords(2,(idetector-1)*nvertices+1:idetector*nvertices) - (iy-0.5_p)
                    weight = real(abs(intersection_polygon_unity_square(polygon, nvertices)), kind=sp)
                    if (weight == 0) cycle
                    if (iroi <= npixels_per_sample) then
                        pmatrix(iroi,idetector)%pixel  = ix - 1 + (iy - 1) * nx
                        pmatrix(iroi,idetector)%weight = weight
                    end if
                    iroi = iroi + 1

                end do

            end do

            ! fill the rest of the pointing matrix
            pmatrix(iroi:,idetector)%pixel  = -1
            pmatrix(iroi:,idetector)%weight = 0
            nroi = max(nroi, iroi-1)

        end do

    end subroutine roi2pmatrix


end module module_pointingmatrix
