! Copyright 2010-2011 Pierre Chanial
! All rights reserved
!
module module_wcs

    use iso_c_binding
    use iso_fortran_env,  only : ERROR_UNIT
    use module_fitstools, only : FLEN_VALUE, ft_read_keyword
    use module_math_old,  only : DEG2RAD, RAD2DEG, distance_2d, neq_real
    use module_string,    only : strinteger, strlowcase, strupcase
    use module_tamasis,   only : p
    implicit none
    private

    type Astrometry
        integer          :: naxis(2)
        real(p)          :: crpix(2), crval(2), cd(2,2)
        character(len=8) :: ctype(2), cunit(2)
    end type Astrometry

    public :: Astrometry
    public :: init_astrometry
    public :: print_astrometry
    public :: cd_info
    public :: projection_scale
    public :: init_gnomonic
    public :: ad2xy_gnomonic
    public :: ad2xys_gnomonic
    public :: init_rotation
    public :: xy2xy_rotation
    public :: refpix_area


contains


    subroutine init_astrometry(header, astr, status)

        character(len=*), intent(in)            :: header
        type(Astrometry), intent(out), optional :: astr
        integer, intent(out)                    :: status

        type(Astrometry)          :: myastr
        logical                   :: found
        integer                   :: has_cd
        real(p)                   :: crota2, cdelt1, cdelt2
        character(len=FLEN_VALUE) :: buffer

        has_cd = 0

        call ft_read_keyword(header, 'naxis1', myastr%naxis(1), status=status)
        if (status /= 0) return

        call ft_read_keyword(header, 'naxis2', myastr%naxis(2), status=status)
        if (status /= 0) return

        call ft_read_keyword(header, 'crpix1', myastr%crpix(1), status=status)
        if (status /= 0) return

        call ft_read_keyword(header, 'crpix2', myastr%crpix(2), status=status)
        if (status /= 0) return

        call ft_read_keyword(header, 'crval1', myastr%crval(1), status=status)
        if (status /= 0) return

        call ft_read_keyword(header, 'crval2', myastr%crval(2), status=status)
        if (status /= 0) return

        call ft_read_keyword(header, 'cd1_1', myastr%cd(1,1), found, status)
        if (status /= 0) return
        if (found) has_cd = has_cd + 1

        call ft_read_keyword(header, 'cd2_1', myastr%cd(2,1), found, status)
        if (status /= 0) return
        if (found) has_cd = has_cd + 1

        call ft_read_keyword(header, 'cd1_2', myastr%cd(1,2), found, status)
        if (status /= 0) return
        if (found) has_cd = has_cd + 1

        call ft_read_keyword(header, 'cd2_2', myastr%cd(2,2), found, status)
        if (status /= 0) return
        if (found) has_cd = has_cd + 1

        call ft_read_keyword(header, 'ctype1', buffer, status=status)
        if (status /= 0) return
        myastr%ctype(1) = strupcase(buffer(1:8))

        call ft_read_keyword(header, 'ctype2', buffer, status=status)
        if (status /= 0) return
        myastr%ctype(2) = strupcase(buffer(1:8))

        call ft_read_keyword(header, 'cunit1', buffer, found, status=status)
        if (status /= 0) return
        if (found) then
            myastr%cunit(1) = strlowcase(buffer(1:8))
        else
            myastr%cunit(1) = 'deg'
        end if

        call ft_read_keyword(header, 'cunit2', buffer, found, status=status)
        if (status /= 0) return
        if (found) then
            myastr%cunit(2) = strlowcase(buffer(1:8))
        else
            myastr%cunit(2) = 'deg'
        end if

        if (has_cd /= 0 .and. has_cd /= 4) then
            write (ERROR_UNIT,'(a)') 'Header has incomplete CD matrix.'
            status = 1
            return
        end if

        if (has_cd == 0) then
            call ft_read_keyword(header, 'cdelt1', cdelt1, found, status)
            if (status /= 0) return
            if (.not. found) then
                write (ERROR_UNIT,'(a)') 'Astrometry definition cannot be extracted from header.'
                status = 1
                return
            end if
            call ft_read_keyword(header, 'cdelt2', cdelt2, found, status)
            if (status /= 0) return
            if (.not. found) then
                write (ERROR_UNIT,'(a)') 'Astrometry definition cannot be extracted from header.'
                status = 1
                return
            end if
            call ft_read_keyword(header, 'crota2', crota2, found, status)
            if (status /= 0) return
            if (.not. found) then
                crota2 = 0.
            end if
            
            crota2 = crota2 * DEG2RAD
            myastr%cd(1,1) =  cos(crota2) * cdelt1
            myastr%cd(2,1) = -sin(crota2) * cdelt1
            myastr%cd(1,2) =  sin(crota2) * cdelt2
            myastr%cd(2,2) =  cos(crota2) * cdelt2

        end if

        if (myastr%ctype(1) == 'RA---TAN' .and. myastr%ctype(2) == 'DEC--TAN') then
            call init_gnomonic(myastr)
        else
            write (ERROR_UNIT,'(a)') "Type '" // myastr%ctype(1) // "', '" // myastr%ctype(2) // "' is not implemented."
            status = 1
            return
        end if

        call init_rotation(myastr)

        if (present(astr)) astr = myastr

    end subroutine init_astrometry


    !-------------------------------------------------------------------------------------------------------------------------------


    subroutine print_astrometry(astr)

        type(Astrometry), intent(in) :: astr

        integer :: naxis, i

        naxis = 0
        do i=1, size(astr%naxis)
            if (astr%naxis(i) /= 0) naxis = naxis + 1
        end do

        write (*,*) 'NAXIS: ', strinteger(naxis), ' (', strinteger(astr%naxis(1)), ',', strinteger(astr%naxis(2)),')'
        write (*,*) 'CRPIX: ', astr%crpix
        write (*,*) 'CRVAL: ', astr%crval
        write (*,*) 'CD   : ', astr%cd(1,:)
        write (*,*) '       ', astr%cd(2,:)
        write (*,*) 'CUNIT: ', astr%cunit(1), ', ', astr%cunit(2)
        write (*,*) 'CTYPE: ', astr%ctype(1), ', ', astr%ctype(2)

    end subroutine print_astrometry


    !-------------------------------------------------------------------------------------------------------------------------------


    subroutine cd_info(cd, cdelt, crota2)

        real(p), intent(in)  :: cd(2,2)
        real(p), intent(out) :: cdelt(2)
        real(p), intent(out) :: crota2
        
        real(p) :: det

        det = cd(1,1) * cd(2,2) - cd(2,1) * cd(1,2)
        cdelt(1) = sign(sqrt(cd(1,1)**2 + cd(1,2)**2), det)
        cdelt(2) = sqrt(cd(2,2)**2 + cd(2,1)**2)
        crota2   = atan2(-cd(2,1),cd(2,2)) * RAD2DEG

    end subroutine cd_info


    !-------------------------------------------------------------------------------------------------------------------------------


    subroutine init_gnomonic(astr)

        type(Astrometry), intent(in) :: astr

        real(p) :: lambda0                ! crval(1) in rad
        real(p) :: phi1, cosphi1, sinphi1 ! cos and sin of crval(2)
        common /gnomonic/ lambda0, cosphi1, sinphi1

        lambda0 = astr%crval(1) * DEG2RAD
        phi1 = astr%crval(2) * DEG2RAD
        cosphi1 = cos(phi1)
        sinphi1 = sin(phi1)

    end subroutine init_gnomonic


    !-------------------------------------------------------------------------------------------------------------------------------


    elemental subroutine ad2xy_gnomonic(a, b)

        real(p), intent(inout) :: a, b

        real(p)                :: lambda, phi, invcosc, xsi, eta
        real(p)                :: lambda0          ! crval[0] in rad
        real(p)                :: cosphi1, sinphi1 ! cos and sin of crval[1]
        common /gnomonic/ lambda0, cosphi1, sinphi1

        lambda = a * DEG2RAD
        phi = b * DEG2RAD
        invcosc = RAD2DEG / (sinphi1*sin(phi)+cosphi1*cos(phi)*cos(lambda-lambda0))
        xsi = invcosc * cos(phi)*sin(lambda-lambda0)
        eta = invcosc * (cosphi1*sin(phi)-sinphi1*cos(phi)*cos(lambda-lambda0))
        call xy2xy_rotation(xsi, eta, a, b)

    end subroutine ad2xy_gnomonic


    !-------------------------------------------------------------------------------------------------------------------------------


    pure subroutine ad2xys_gnomonic(ad, x, y, scale)

        real(p), intent(in)  :: ad(:,:)          ! R.A. and declination in degrees
        real(p), intent(out) :: x(size(ad,2))
        real(p), intent(out) :: y(size(ad,2))
        real(p), intent(out) :: scale(size(ad,2))

        real(p) :: lambda, phi, invcosc, xsi, eta
        integer :: i
        real(p) :: lambda0          ! crval[0] in rad
        real(p) :: cosphi1, sinphi1 ! cos and sin of crval[1]

        common /gnomonic/ lambda0, cosphi1, sinphi1

        do i = 1, size(ad,2)
            lambda = ad(1,i) * DEG2RAD
            phi = ad(2,i) * DEG2RAD
            invcosc = 1 / (sinphi1*sin(phi)+cosphi1*cos(phi)*cos(lambda-lambda0))
            xsi = RAD2DEG * invcosc * cos(phi)*sin(lambda-lambda0)
            eta = RAD2DEG * invcosc * (cosphi1*sin(phi)-sinphi1*cos(phi)*cos(lambda-lambda0))

            call xy2xy_rotation(xsi, eta, x(i), y(i))
            scale(i) = invcosc ** 3
        end do

    end subroutine ad2xys_gnomonic


    !-------------------------------------------------------------------------------------------------------------------------------


    elemental subroutine ad2xy_gnomonic_vect(a, b)

        real(p), intent(inout) :: a, b ! R.A. and declination in degrees on input, x,y on output

        real(p)             :: lambda, phi, invcosc, xsi, eta
        real(p)             :: lambda0          ! crval[0] in rad
        real(p)             :: cosphi1, sinphi1 ! cos and sin of crval[1]
        common /gnomonic/ lambda0, cosphi1, sinphi1

        lambda  = a * DEG2RAD
        phi     = b * DEG2RAD
        invcosc = RAD2DEG / (sinphi1*sin(phi)+cosphi1*cos(phi)*cos(lambda-lambda0))
        xsi = invcosc * cos(phi)*sin(lambda-lambda0)
        eta = invcosc * (cosphi1*sin(phi)-sinphi1*cos(phi)*cos(lambda-lambda0))
        call xy2xy_rotation(xsi, eta, a, b)

    end subroutine ad2xy_gnomonic_vect


    !-------------------------------------------------------------------------------------------------------------------------------


    subroutine projection_scale(astr, array, status)
        
        type(Astrometry), intent(in) :: astr
        real(p), intent(out)         :: array(:,:)
        integer, intent(out)         :: status

        real(p) :: cdelt(2), rot

        call cd_info(astr%cd, cdelt, rot)

        if (astr%ctype(1) == 'RA---TAN' .and. astr%ctype(2) == 'DEC--TAN') then
            if (neq_real(abs(cdelt(1)), abs(cdelt(2)), 1.e-7_p)) then
                write (ERROR_UNIT,'(a)') 'The scaling of a map with inhomogeneous CDELT is not implemented.'
                status = 1
                return
            end if
            call projection_scale_gnomonic(array, astr%crpix, cdelt)
        else
            write (ERROR_UNIT,'(a)') "Scaling for projection type '" // astr%ctype(1) // "', '" // astr%ctype(2) // "' is not imple&
                  &mented."
            status = 1
            return
        end if

        status = 0

    end subroutine projection_scale


    !-------------------------------------------------------------------------------------------------------------------------------


    subroutine projection_scale_gnomonic(array, origin, resolution)

        real(p), intent(out) :: array(:,:)
        real(p), intent(in)  :: origin(2)
        real(p), intent(in)  :: resolution(2)

        call distance_2d(array, origin - 1, tan(resolution * DEG2RAD))

        !$omp parallel workshare
        array = cos(atan(array))**3
        !$omp end parallel workshare

    end subroutine projection_scale_gnomonic


    !-------------------------------------------------------------------------------------------------------------------------------


    subroutine init_rotation(astr)

        type(Astrometry), intent(in) :: astr

        real(p) :: cdinv(2,2), crpix(2)
        common /rotation/ cdinv, crpix

        cdinv = reshape([astr%cd(2,2), -astr%cd(2,1), -astr%cd(1,2), astr%cd(1,1)], [2,2]) / &
                (astr%cd(1,1)*astr%cd(2,2) - astr%cd(2,1)*astr%cd(1,2))
        ! Convention is (0,0) for the bottom left pixel, unlike the FITS convention.
        crpix = astr%crpix - 1

    end subroutine init_rotation


    !-------------------------------------------------------------------------------------------------------------------------------


    elemental subroutine xy2xy_rotation(xsi, eta, x, y)

        real(p), intent(in)  :: xsi, eta
        real(p), intent(out) :: x, y

        real(p) :: cdinv(2,2), crpix(2)
        common /rotation/ cdinv, crpix

        x = cdinv(1,1)*xsi + cdinv(1,2)*eta + crpix(1)
        y = cdinv(2,1)*xsi + cdinv(2,2)*eta + crpix(2)

    end subroutine xy2xy_rotation



    !-------------------------------------------------------------------------------------------------------------------------------


    function refpix_area()

        real(p) :: refpix_area

        real(p) :: cdinv(2,2), crpix(2)
        common /rotation/ cdinv, crpix
        
        refpix_area = 1 / abs(cdinv(1,1) * cdinv(2,2) - cdinv(1,2) * cdinv(2,1)) * 3600._p**2

    end function refpix_area


 end module module_wcs
