module wcsutils

    implicit none

    integer, parameter :: p = kind(1.d0)
    real(p), parameter :: PI = 4_p * atan(1._p)
    real(p), parameter :: DEG2RAD = PI / 180._p
    real(p), parameter :: RAD2DEG = 180._p / PI
    real(p), parameter ::                                                                                                          &
        NaN  = transfer('1111111111111000000000000000000000000000000000000000000000000000'b, 0._p),                                &
        mInf = transfer('1111111111110000000000000000000000000000000000000000000000000000'b, 0._p),                                &
        pInf = transfer('0111111111110000000000000000000000000000000000000000000000000000'b, 0._p)

contains

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


end module wcsutils
