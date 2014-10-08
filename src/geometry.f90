module geometry

    use module_tamasis, only : p
    use module_math_old, only : DEG2RAD, RAD2DEG
    implicit none


contains


    subroutine create_grid(m, n, spacing, xreflection, yreflection, angle, xcenter, ycenter, coords)
        !f2py threadsafe
        integer, intent(in)   :: m                ! number of rows
        integer, intent(in)   :: n                ! number of columns
        real*8, intent(in)    :: spacing          ! size of the detector placeholder
        logical, intent(in)   :: xreflection      ! reflection along the x-axis (before rotation)
        logical, intent(in)   :: yreflection      ! reflection along the y-axis (before rotation)
        real*8, intent(in)    :: angle            ! counter-clockwise rotation angle in degrees (before translation)
        real*8, intent(in)    :: xcenter, ycenter ! coordinates of the grid center
        real*8, intent(inout) :: coords(2,n,m)    ! output coordinates of the detector corners (first dimension is x and y)

        integer :: i, j
        real*8  :: x, y, x0, y0, r11, r12, r21, r22

        r11 = cos(DEG2RAD * angle)
        r21 = sin(DEG2RAD * angle)
        r12 = -r21
        r22 = r11
        if (xreflection) then
            r11 = -r11
            r21 = -r21
        end if
        if (.not. yreflection) then
            r12 = -r12
            r22 = -r22
        end if
        x0 = -0.5_p * (n + 1)
        y0 = -0.5_p * (m + 1)
        
        !$omp parallel do private(x, y)
        do j = 1, m
            y = (y0 + j) * spacing
            do i = 1, n
                x = (x0 + i) * spacing
                coords(1,i,j) = xcenter + r11 * x + r12 * y
                coords(2,i,j) = ycenter + r21 * x + r22 * y
            end do
        end do
        !$omp end parallel do
        
    end subroutine create_grid


    !-------------------------------------------------------------------------------------------------------------------------------


    subroutine create_grid_squares(m, n, spacing, filling_factor, xreflection, yreflection, angle, xcenter, ycenter, coords)
        !f2py threadsafe
        integer, intent(in)   :: m                ! number of rows
        integer, intent(in)   :: n                ! number of columns
        real*8, intent(in)    :: spacing          ! size of the detector placeholder
        real*8, intent(in)    :: filling_factor   ! fraction of transmitting detector area
        logical, intent(in)   :: xreflection      ! reflection along the x-axis (before rotation)
        logical, intent(in)   :: yreflection      ! reflection along the y-axis (before rotation)
        real*8, intent(in)    :: angle            ! counter-clockwise rotation angle in degrees (before translation)
        real*8, intent(in)    :: xcenter, ycenter ! coordinates of the grid center
        real*8, intent(inout) :: coords(2,4,n,m)  ! output coordinates of the detector corners (first dimension is x and y)

        integer :: i, j, k
        real*8 :: x, y, x0, y0, size_eff, r11, r12, r21, r22, tmp

        size_eff = spacing * sqrt(filling_factor)
        r11 = cos(DEG2RAD * angle)
        r21 = sin(DEG2RAD * angle)
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
        x0 = -0.5_p * ((n + 1) * spacing + size_eff)
        y0 =  0.5_p * ((m + 1) * spacing - size_eff)
        
        !$omp parallel do private(x, y, tmp)
        do j = 1, m
            y = y0 - spacing * j
            do i = 1, n
                x = x0 + spacing * i
                coords(1,1,i,j) = x + size_eff
                coords(2,1,i,j) = y + size_eff
                coords(1,2,i,j) = x
                coords(2,2,i,j) = y + size_eff
                coords(1,3,i,j) = x
                coords(2,3,i,j) = y
                coords(1,4,i,j) = x + size_eff
                coords(2,4,i,j) = y
                do k = 1, 4
                    tmp = coords(1,k,i,j)
                    coords(1,k,i,j) = xcenter + r11 * tmp + r12 * coords(2,k,i,j)
                    coords(2,k,i,j) = ycenter + r21 * tmp + r22 * coords(2,k,i,j)
                end do
            end do
        end do
        !$omp end parallel do
        
    end subroutine create_grid_squares


    !-------------------------------------------------------------------------------------------------------------------------------


    subroutine rotate_2d(x, y, n, angle)
        !f2py threadsafe
        real*8, intent(in)    :: x(2,n)
        real*8, intent(inout) :: y(2,n)
        real*8, intent(in)    :: angle
        integer, intent(in)   :: n

        integer :: i
        real*8  :: sinangle, cosangle

        sinangle = sin(angle * DEG2RAD)
        cosangle = cos(angle * DEG2RAD)
        !$omp parallel do
        do i = 1, n
            y(1,i) = cosangle * x(1,i) - sinangle * x(2,i)
            y(2,i) = sinangle * x(1,i) + cosangle * x(2,i)
        end do
        !$omp end parallel do

    end subroutine rotate_2d


    !-------------------------------------------------------------------------------------------------------------------------------


    subroutine rotate_2d_inplace(x, n, angle)
        !f2py threadsafe
        real*8, intent(inout) :: x(2,n)
        real*8, intent(in)    :: angle
        integer, intent(in)   :: n

        integer :: i
        real*8  :: tmp, sinangle, cosangle

        sinangle = sin(angle * DEG2RAD)
        cosangle = cos(angle * DEG2RAD)
        !$omp parallel do private(tmp)
        do i = 1, n
            tmp = x(1,i)
            x(1,i) = cosangle * x(1,i) - sinangle * x(2,i)
            x(2,i) = sinangle * tmp    + cosangle * x(2,i)
        end do
        !$omp end parallel do

    end subroutine rotate_2d_inplace


    !-------------------------------------------------------------------------------------------------------------------------------


    subroutine surface_simple_polygon(xy, output, nvertices, npolygons)
        !f2py threadsafe
        integer*8, intent(in) :: nvertices, npolygons
        real*8, intent(in)    :: xy(2, nvertices, npolygons)
        real*8, intent(inout) :: output(npolygons)
        integer*8             :: i, j, k

        !$omp parallel do private(j)
        do k=1, npolygons
            output(k) = 0
            j = nvertices
            do i=1, nvertices
                output(k) = output(k) + xy(1,j,k)*xy(2,i,k) - xy(2,j,k)*xy(1,i,k)
                j = i
            end do
            output(k) = 0.5_p * output(k)
        end do
        !$omp end parallel do

    end subroutine surface_simple_polygon


end module geometry
