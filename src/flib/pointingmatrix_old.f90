module pointingmatrix

    use module_tamasis, only : p
    use module_pointingmatrix, backprojection_weight_ => backprojection_weight,&
                               roi2pmatrix_cartesian_ => roi2pmatrix_cartesian
    implicit none

contains
    
    subroutine backprojection_weight(matrix, data, map1d, weight1d, npixels_per_sample, nsamples, npixels)

        integer*8, intent(in)             :: nsamples
        !f2py integer*8, intent(in)       :: matrix(npixels_per_sample*nsamples)
        type(PointingElement), intent(in) :: matrix(npixels_per_sample,nsamples)
        real(p), intent(in)               :: data(nsamples)
        real(p), intent(inout)            :: map1d(npixels)
        real(p), intent(inout)            :: weight1d(npixels)
        integer, intent(in)               :: npixels_per_sample
        integer, intent(in)               :: npixels

        call backprojection_weight_(matrix, data, map=map1d, weight=weight1d)

    end subroutine backprojection_weight


    !---------------------------------------------------------------------------


    subroutine backprojection_weight_mask(matrix, data, mask, map1d, weight1d, npixels_per_sample, nsamples, npixels)

        integer*8, intent(in)             :: nsamples
        !f2py integer*8,intent(in)        :: matrix(npixels_per_sample*nsamples)
        type(PointingElement), intent(in) :: matrix(npixels_per_sample,nsamples)
        real(p), intent(in)               :: data(nsamples)
        logical*1, intent(in)             :: mask(nsamples)
        real(p), intent(inout)            :: map1d(npixels)
        real(p), intent(inout)            :: weight1d(npixels)
        integer, intent(in)               :: npixels_per_sample
        integer, intent(in)               :: npixels

        call backprojection_weight_(matrix, data, mask, map1d, weight1d)

    end subroutine backprojection_weight_mask


    !---------------------------------------------------------------------------


    subroutine direct(matrix, map1d, signal, npixels_per_sample, nsamples, npixels)

        integer*8, intent(in)                :: nsamples
        !f2py integer*8, intent(inout)       :: matrix(npixels_per_sample*nsamples)
        type(PointingElement), intent(inout) :: matrix(npixels_per_sample,nsamples)
        real(p), intent(in)    :: map1d(npixels)
        real(p), intent(inout) :: signal(nsamples)
        integer, intent(in)    :: npixels_per_sample
        integer, intent(in)    :: npixels

        if (npixels_per_sample == 1) then
            call pmatrix_direct_one_pixel_per_sample(matrix(1,:), map1d, signal)
        else
            call pmatrix_direct(matrix, map1d, signal)
        end if

    end subroutine direct


    !---------------------------------------------------------------------------


    subroutine intersects(matrix, pixel, npixels_per_sample, nsamples, ndetectors, out)

        integer*8, intent(in)                   :: nsamples
        !f2py integer*8, intent(in)             :: matrix(npixels_per_sample*nsamples*ndetectors)
        type(PointingElement), intent(in)       :: matrix(npixels_per_sample,nsamples,ndetectors)
        integer(kind(matrix%index)), intent(in) :: pixel
        integer, intent(in)                     :: npixels_per_sample
        integer, intent(in)                     :: ndetectors
        logical*1, intent(out)                  :: out
        
        out = any(matrix%index == pixel)

    end subroutine intersects


    !---------------------------------------------------------------------------


    subroutine intersects_openmp2(matrix, pixel, npixels_per_sample, nsamples, ndetectors, out)

        integer*8, intent(in)                   :: nsamples
        !f2py integer*8, intent(in)             :: matrix(npixels_per_sample*nsamples*ndetectors)
        type(PointingElement), intent(in)       :: matrix(npixels_per_sample,nsamples,ndetectors)
        integer(kind(matrix%index)), intent(in) :: pixel
        integer, intent(in)                     :: npixels_per_sample
        integer, intent(in)                     :: ndetectors
        logical*1, intent(out)                  :: out
        
        !$omp parallel workshare
        out = any(matrix%index == pixel)
        !$omp end parallel workshare

    end subroutine intersects_openmp2


    !---------------------------------------------------------------------------


    subroutine intersects_axis2(matrix, pixel, npixels_per_sample, nsamples, ndetectors, out)

        integer*8, intent(in)                   :: nsamples
        !f2py integer*8, intent(in)             :: matrix(npixels_per_sample*nsamples*ndetectors)
        type(PointingElement), intent(in)       :: matrix(npixels_per_sample,nsamples,ndetectors)
        integer(kind(matrix%index)), intent(in) :: pixel
        integer, intent(in)                     :: npixels_per_sample
        integer, intent(in)                     :: ndetectors
        logical*1, intent(out)                  :: out(ndetectors)

        integer   :: idetector, ipixel
        integer*8 :: isample

        out = .false.
        !$omp parallel do schedule(guided)
        loop_detector: do idetector = 1, ndetectors
            do isample = 1, nsamples
                do ipixel = 1, npixels_per_sample
                    if (matrix(ipixel,isample,idetector)%index == pixel) then
                        out(idetector) = .true.
                        cycle loop_detector
                    end if
                end do
            end do
        end do loop_detector
        !$omp end parallel do

    end subroutine intersects_axis2


    !---------------------------------------------------------------------------


    subroutine intersects_axis3(matrix, pixel, npixels_per_sample, nsamples, ndetectors, out)

        integer*8, intent(in)                   :: nsamples
        !f2py integer*8, intent(in)             :: matrix(npixels_per_sample*nsamples*ndetectors)
        type(PointingElement), intent(in)       :: matrix(npixels_per_sample,nsamples,ndetectors)
        integer(kind(matrix%index)), intent(in) :: pixel
        integer, intent(in)                     :: npixels_per_sample
        integer, intent(in)                     :: ndetectors
        logical*1, intent(out)                  :: out(nsamples)

        integer   :: idetector, ipixel
        integer*8 :: isample

        out = .false.
        !$omp parallel do schedule(guided)
        loop_sample: do isample = 1, nsamples
            do idetector = 1, ndetectors
                do ipixel = 1, npixels_per_sample
                    if (matrix(ipixel,isample,idetector)%index == pixel) then
                        out(isample) = .true.
                        cycle loop_sample
                    end if
                end do
            end do
        end do loop_sample
        !$omp end parallel do

    end subroutine intersects_axis3


    !---------------------------------------------------------------------------


    subroutine isvalid(matrix, npixels_per_sample, nsamples, npixels, output)

        !f2py threadsafe
        integer*8, intent(in)             :: nsamples
        !f2py integer*8                   :: matrix(npixels_per_sample*nsamples)
        type(PointingElement), intent(in) :: matrix(npixels_per_sample,nsamples)
        integer, intent(in)               :: npixels_per_sample
        integer, intent(in)               :: npixels
        logical*1, intent(out)            :: output
        integer :: index, ipixel, isample

        output = .false.
        do isample = 1, size(matrix, 2)
            do ipixel = 1, size(matrix, 1)
                index = matrix(ipixel,isample)%index
                if (index == -1) cycle
                if (index < 0 .or. index >= npixels) return
            end do
        end do
        output = .true.

    end subroutine isvalid


    !---------------------------------------------------------------------------


    subroutine mask(matrix, mask1d, npixels_per_sample, nsamples, npixels)

        integer*8, intent(in)             :: nsamples
        !f2py integer*8, intent(in)       :: matrix(npixels_per_sample*nsamples)
        type(PointingElement), intent(in) :: matrix(npixels_per_sample,nsamples)
        logical*1, intent(inout)          :: mask1d(npixels)
        integer, intent(in)               :: npixels_per_sample
        integer, intent(in)               :: npixels

        call pmatrix_mask(matrix, mask1d)

    end subroutine mask


    !---------------------------------------------------------------------------
    
    
    subroutine pack(matrix, mask1d, npixels_per_sample, nsamples, npixels)

        integer*8, intent(in)                :: nsamples
        !f2py integer*8, intent(inout)       :: matrix(npixels_per_sample*nsamples)
        type(PointingElement), intent(inout) :: matrix(npixels_per_sample,nsamples)
        logical*1, intent(in)                :: mask1d(npixels)
        integer, intent(in)                  :: npixels_per_sample
        integer, intent(in)                  :: npixels

        call pmatrix_pack(matrix, mask1d)

    end subroutine pack


    !---------------------------------------------------------------------------


    subroutine ptp(matrix, array, npixels_per_sample, nsamples, npixels)

        integer*8, intent(in)                :: nsamples
        !f2py integer*8, intent(inout)       :: matrix(npixels_per_sample*nsamples)
        type(PointingElement), intent(inout) :: matrix(npixels_per_sample,nsamples)
        real(p), intent(inout)               :: array(npixels, npixels)
        integer, intent(in)                  :: npixels_per_sample
        integer, intent(in)                  :: npixels

        call pmatrix_ptp(matrix, array)

    end subroutine ptp


    !---------------------------------------------------------------------------


    subroutine ptp_one(matrix, array, nsamples, npixels)

        integer*8, intent(in)                :: nsamples
        !f2py integer*8, intent(inout)       :: matrix(nsamples)
        type(PointingElement), intent(inout) :: matrix(nsamples)
        real(p), intent(inout)               :: array(0:npixels-1)
        integer, intent(in)                  :: npixels
        integer*8 :: isample

        do isample = 1, nsamples
            array(matrix(isample)%index) = array(matrix(isample)%index) +    &
                                           matrix(isample)%value**2
        end do

    end subroutine ptp_one


    !---------------------------------------------------------------------------


    subroutine transpose(matrix, signal, map1d, npixels_per_sample, nsamples, npixels)

        integer*8, intent(in)                :: nsamples
        !f2py integer*8, intent(inout)       :: matrix(npixels_per_sample*nsamples)
        type(PointingElement), intent(inout) :: matrix(npixels_per_sample,nsamples)
        real(p), intent(in)                  :: signal(nsamples)
        real(p), intent(inout)               :: map1d(npixels)
        integer, intent(in)                  :: npixels_per_sample
        integer, intent(in)                  :: npixels

        if (npixels_per_sample == 1) then
            call pmatrix_transpose_one_pixel_per_sample(matrix(1,:), signal, map1d)
        else
            call pmatrix_transpose(matrix, signal, map1d)
        end if

    end subroutine transpose


    !---------------------------------------------------------------------------


    subroutine roi2pmatrix_cartesian(roi, coords, ncoords, nvertices,          &
        npixels_per_sample, nx, ny, matrix, new_npixels_per_sample, out)

        !f2py threadsafe
        integer, intent(in)                :: roi(2,2,ncoords)
        real(p), intent(in)                :: coords(2,nvertices,ncoords)
        integer, intent(in)                :: ncoords, nvertices
        integer, intent(in)                :: npixels_per_sample
        integer, intent(in)                :: nx, ny
        integer, intent(out)               :: new_npixels_per_sample
        logical, intent(out)               :: out
        !f2py integer*8, intent(in)        :: matrix(npixels_per_sample*ncoords)
        type(PointingElement),intent(inout):: matrix(npixels_per_sample,ncoords)

        new_npixels_per_sample = 0
        call roi2pmatrix_cartesian_(roi, coords, nx, ny, new_npixels_per_sample, out, matrix)

    end subroutine roi2pmatrix_cartesian


end module pointingmatrix
