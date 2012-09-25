module pointingmatrix
    
    implicit none

    integer, parameter :: p = kind(1.d0)
    integer, parameter :: sp = kind(1.)

    type PointingElement
        real*4  :: value
        integer :: index
    end type PointingElement

contains

    subroutine check(matrix, npixels_per_sample, nsamples, npixels, isvalid)
        !f2py threadsafe
        !f2py integer*8, dimension(npixels_per_sample*nsamples) :: matrix

        type(PointingElement), intent(in) :: matrix(npixels_per_sample,nsamples)
        integer, intent(in)               :: npixels_per_sample
        integer, intent(in)               :: nsamples
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


    subroutine roi2matrix_cartesian(roi, coords, ncoords, nvertices,           &
        npixels_per_sample, nx, ny, matrix, new_npixels_per_sample, out)
        !f2py threadsafe
        !f2py integer*8, dimension(npixels_per_sample*ncoords) :: matrix

        integer, intent(in)                 :: roi(2,2,ncoords)
        real*8, intent(in)                  :: coords(2,nvertices,ncoords)
        integer, intent(in)                 :: ncoords, nvertices
        integer, intent(in)                 :: npixels_per_sample
        integer, intent(in)                 :: nx, ny
        integer, intent(out)                :: new_npixels_per_sample
        logical, intent(out)                :: out
        type(PointingElement), intent(inout)::matrix(npixels_per_sample,ncoords)

        real*8  :: polygon(2,nvertices)
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
        logical :: neg_delta_x! TRUE if delta_x < 0

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


end module pointingmatrix
