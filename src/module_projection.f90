! Copyright 2010-2011 Pierre Chanial
! All rights reserved
!
module module_projection

    use module_sort,    only : qsortgi
    use module_stack,   only : stack_int
    use module_tamasis, only : p
    implicit none
    private

    public :: intersection_polygon_unity_square
    public :: intersection_segment_unity_square
    public :: surface_parallelogram
    public :: point_in_polygon
    public :: convex_hull
    public :: set_pivot
    public :: qsorti_point

    real(p), private          :: pivot_(2)
    real(p), pointer, private :: array_point(:,:)
    !$omp threadprivate(pivot_, array_point)

contains


    pure function intersection_polygon_unity_square(xy, nvertices) result(output)

        real(p)             :: output
        real(p), intent(in) :: xy(2,nvertices)
        integer, intent(in) :: nvertices

        integer             :: i, j

        output = 0
        j = nvertices
        do i=1, nvertices
            output = output + intersection_segment_unity_square(xy(1,i), xy(2,i), xy(1,j), xy(2,j))
            j = i
        end do

    end function intersection_polygon_unity_square


    !-------------------------------------------------------------------------------------------------------------------------------


    pure function intersection_segment_unity_square(x1,  y1, x2,  y2) result(output)

        real(p)             :: output
        real(p), intent(in) :: x1, y1, x2, y2 ! first and second point coordinates

        ! we will use the following variables :

        real(p) :: pente               ! The slope of the straight line going through p1, p2
        real(p) :: ordonnee            ! The point where the straight line crosses y-axis
        real(p) :: delta_x             ! = x2-x1
        real(p) :: xmin, xmax          ! minimum and maximum values of x to consider
                                       ! (clipped in the square (0,0),(1,0),(1,1),(0,1)
        real(p) :: ymin, ymax          ! minimum and maximum values of y to consider
                                       ! (clipped in the square (0,0),(1,0),(1,1),(0,1)
        real(p) :: xhaut               ! value of x at which straight line crosses the
                                       ! (0,1),(1,1) line
        logical :: negative_delta_x    ! TRUE if delta_x < 0

        ! Check for vertical line : the area intercepted is 0
        if (x1 == x2) then
            output = 0
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
            output = 0
            return    ! outside, the area is 0
        end if

        ! We compute xmin, xmax, clipped between 0 and 1 in x
        ! then we compute pente (slope) and ordonnee and use it to get ymin
        ! and ymax
        xmin = max(xmin, 0._p)
        xmax = min(xmax, 1._p)

        delta_x = x2 - x1
        negative_delta_x = delta_x < 0
        pente = (y2 - y1) / delta_x
        ordonnee  = y1 - pente * x1
        ymin = pente * xmin + ordonnee
        ymax = pente * xmax + ordonnee

        ! Trap segment entirely below axis
        if (ymin < 0 .and. ymax < 0) then
            output = 0
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
            if (negative_delta_x) then
                output = xmin - xmax
            else
                output = xmax - xmin
            end if
            return

        end if

        if (ymin <= 1 .and. ymax <= 1) then
          ! Segment is entirely within square
          if (negative_delta_x) then
             output = 0.5_p * (xmin-xmax) * (ymax+ymin)
          else
             output = 0.5_p * (xmax-xmin) * (ymax+ymin)
          end if
          return
        end if

        ! otherwise it must cross the top of the square
        ! the crossing occurs at xhaut
        xhaut = (1 - ordonnee) / pente
        !!if ((xhaut < xmin) .or. (xhaut > xmax))   cout << " BUGGGG "
        if (ymin < 1) then
            if (negative_delta_x) then
                output = -(0.5_p * (xhaut-xmin) * (1+ymin) + xmax - xhaut)
            else
                output = 0.5_p * (xhaut-xmin) * (1+ymin) + xmax - xhaut
            end if
        else
            if (negative_delta_x) then
                output = -(0.5_p * (xmax-xhaut) * (1+ymax) + xhaut-xmin)
            else
                output = 0.5_p * (xmax-xhaut) * (1+ymax) + xhaut-xmin
            end if
        end if

    end function intersection_segment_unity_square


    !-------------------------------------------------------------------------------------------------------------------------------


    ! returns signed surface of a parallelogram
    ! surface = det( (b-a, c-a) )
    ! positive if a, b, c are counter-clockwise, negative otherwise
    pure function point_on_or_left(a, b, c)

        real(p), intent(in) :: a(:), b(:), c(:)
        logical             :: point_on_or_left

        point_on_or_left = surface_parallelogram(a, b, c) >= 0

    end function point_on_or_left


    !-------------------------------------------------------------------------------------------------------------------------------


    ! returns signed surface of a parallelogram
    ! surface = det( (b-a, c-a) )
    ! positive if a, b, c are counter-clockwise, negative otherwise
    pure function surface_parallelogram(a, b, c)

        real(p)             :: surface_parallelogram
        real(p), intent(in) :: a(:), b(:), c(:)

        surface_parallelogram = (b(1) - a(1)) * (c(2) - a(2)) - (c(1) - a(1)) * (b(2) - a(2))

    end function surface_parallelogram


    !-------------------------------------------------------------------------------------------------------------------------------


    ! quick sort for points, using 1) angles 2) length (see function compare below)
    subroutine qsorti_point(array, index)

        real(p), intent(in), target :: array(:,:)
        integer, intent(out)        :: index(size(array,2))

        array_point => array
        call qsortgi(size(array_point,2), compare_point, index)
        array_point => null()

    end subroutine qsorti_point


    !-------------------------------------------------------------------------------------------------------------------------------


    ! compare two points
    ! returns -1,0,+1 if p1 < p2, =, or > respectively;
    ! here "<" means smaller angle.  Follows the conventions of qsort.
    function compare_point(point1, point2)

        integer, intent(in) :: point1, point2
        integer             :: compare_point

        real(p)             :: area, length

        area = surface_parallelogram(pivot_, array_point(:,point1), array_point(:,point2))
        if (area > 0) then
            compare_point = 1
            return
        end if
        if (area < 0) then
            compare_point = -1
            return
        end if
        length = dot_product(array_point(:,point2), array_point(:,point2)) -                                                       &
                 dot_product(array_point(:,point1), array_point(:,point1))
        if (length > 0) then
            compare_point = 1
            return
        endif
        if (length < 0) then
            compare_point = -1
            return
        endif
        compare_point = 0

    end function compare_point


    !-------------------------------------------------------------------------------------------------------------------------------


    ! set pivot in a global variable accessible from compare_point
    subroutine set_pivot(pivot)

        real(p), intent(in) :: pivot(2)

        pivot_ = pivot

    end subroutine set_pivot


    !-------------------------------------------------------------------------------------------------------------------------------


    ! find the rightmost among the bottommost vertices
    function find_pivot(points)

        real(p), intent(in) :: points(:,:)
        integer             :: find_pivot

        integer             :: imin(1)

        imin = minloc(points(1,:), points(2,:) == minval(points(2,:)))
        find_pivot = imin(1)

    end function find_pivot


    !-------------------------------------------------------------------------------------------------------------------------------


    ! returns 1, 0, -1 if a point is inside, on an edge or outside a polygon
    ! algo from Simulation of Simplicity: A Technique to Cope with Degenerate Cases in Geometric Algorithms
    ! modified to cope with the cases in which the point lies on an edge
    function point_in_polygon(point, polygon)

        real(p), intent(in) :: point(2), polygon(:,:)
        integer             :: point_in_polygon

        integer :: i, j, n
        real(p) :: a, b

        point_in_polygon = -1
        n = size(polygon, 2)

        ! first, test if a point is on one of the vertices
        do i = 1, n
            if (all(polygon(:,i) == point)) then
                point_in_polygon = 0
                return
            end if
        end do

        ! loop over the edges. Count how many time a horizontal rightwards ray crosses the polygon
        ! if it crosses an even number of times, the point is outside (Jordan curve theorem).
        j = n
        do i = 1, n
            if ((polygon(2,i) > point(2)) .neqv. (polygon(2,j) > point(2))) then
                a = point(1) - polygon(1,i)
                b = (polygon(1,j) - polygon(1,i)) * (point(2)-polygon(2,i)) / &
                    (polygon(2,j) - polygon(2,i))
                if (a == b) then
                    point_in_polygon = 0
                    return
                end if
                if (a < b) then
                    point_in_polygon = - point_in_polygon;
                end if
            else if (polygon(2,i) == polygon(2,j) .and. point(2) == polygon(2,i) .and. (polygon(1,i) > point(1) .neqv. polygon(1,j)&
                     > point(1))) then
                point_in_polygon = 0
                return
            endif
            j = i
        end do

    end function point_in_polygon


    !-------------------------------------------------------------------------------------------------------------------------------


    ! Graham scan implementation
    subroutine convex_hull(points, index)

        real(p), intent(in)               :: points(:,:)
        integer, intent(out), allocatable :: index(:)

        integer                           :: i, n, ipivot, junk
        integer                           :: isort(size(points,2))
        class(stack_int), allocatable     :: stack

        n = size(points,2)
        ipivot = find_pivot(points)
        call set_pivot(points(:, ipivot))
        call qsorti_point(points, isort)

        ! we start the stack with the last and first point, which by construction are part of the hull
        allocate(stack)
        call stack%push(isort(n))
        call stack%push(isort(1))

        ! loop over the vertices, ensuring that the point N is on the left of the line N-2, N-1
        i = 2
        do while (i <= n)
            if (point_on_or_left(points(:,stack%head%next%value), points(:,stack%head%value), points(:,isort(i)))) then
                if (i == n) exit
                call stack%push(isort(i))
                i = i + 1
            else
                junk = stack%pop()
            end if
        end do

        call stack%to_array(index)
        index = index(size(index):1:-1)

    end subroutine convex_hull


end module module_projection
