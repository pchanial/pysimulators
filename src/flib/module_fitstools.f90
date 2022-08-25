! Copyright 2010-2011 Pierre Chanial
! All rights reserved
!
module module_fitstools

    use iso_c_binding
    use iso_fortran_env,  only : ERROR_UNIT, OUTPUT_UNIT
    use module_precision, only : sp, dp, qp
    use module_string,    only : strinteger, strlowcase, strsection, strupcase
    use module_tamasis,   only : p
    implicit none
    private

    public :: FLEN_KEYWORD
    public :: FLEN_CARD
    public :: FLEN_VALUE
    public :: FLEN_COMMENT
    public :: FLEN_ERRMSG
    public :: FLEN_STATUS
    public :: ft_read_keyword

    integer, private, parameter :: GROUP = 1
    integer, private, parameter :: NULLVAL = 0

    integer, parameter :: FLEN_KEYWORD = 71
    integer, parameter :: FLEN_CARD    = 80
    integer, parameter :: FLEN_VALUE   = 70
    integer, parameter :: FLEN_COMMENT = 72
    integer, parameter :: FLEN_ERRMSG  = 80
    integer, parameter :: FLEN_STATUS  = 30


    interface ft_read_keyword
        module procedure ft_read_keyword_header_logical, ft_read_keyword_header_int4, ft_read_keyword_header_int8,                 &
                         ft_read_keyword_header_real4, ft_read_keyword_header_real8, ft_read_keyword_header_character
    end interface ft_read_keyword



contains


    subroutine get_keyword(header, keyword, value, found, comment, must_exist, status)

        character(len=*), intent(in)                       :: header
        character(len=*), intent(in)                       :: keyword
        character(len=FLEN_VALUE), intent(out)             :: value
        logical, intent(out)                               :: found
        character(len=FLEN_COMMENT), intent(out), optional :: comment
        logical, intent(in), optional                      :: must_exist
        integer, intent(out)                               :: status

        character(len=FLEN_VALUE)   :: buffer
        character(len=FLEN_KEYWORD) :: strlowkeyword
        integer                     :: i, ikeyword, ncards

        status = 0
        value = ' '

        ncards = (len(header)-1) / 80 + 1! allow for an additional \0
        found = .false.
        if (present(comment)) comment = ' '
        strlowkeyword = strlowcase(keyword)

        ! find the record number associated to the keyword
        do ikeyword = 1, ncards
           if (strlowcase(header((ikeyword-1)*80+1:(ikeyword-1)*80+8)) == strlowkeyword) then
              found = .true.
              exit
           end if
        end do

        ! the keyword is not found
        if (.not. found) then
            if (present(must_exist)) then
                if (must_exist) then
                    status = 202
                    write (ERROR_UNIT, '(a)') "Missing keyword '" // strupcase(keyword) // "' in FITS header."
                    return
                end if
            end if
            return
        end if

        buffer = adjustl(header((ikeyword-1)*80+11:ikeyword*80))

        ! simple case, the value is not enclosed in quotes
        if (buffer(1:1) /= "'") then
            i = index(buffer, '/')
            if (i == 0) i = 71
            goto 999 ! end of slash
        end if

        ! find ending quote
        i = 2
        do while (i <= 70)
            if (buffer(i:i) == "'") then
                i = i + 1
                if (i == 71) goto 999 ! end of slash
                if (buffer(i:i) /= "'") exit
            end if
            i = i + 1
        end do

        ! i points right after ending quote, let's find '/' ignoring what's before it
        do while (i <= 70)
            if (buffer(i:i) == '/') exit
            i = i + 1
        end do

        ! i points to '/' or is 71
    999 value = buffer(1:i-1)
        if (present(comment)) comment = buffer(i+1:)

    end subroutine get_keyword


    !-------------------------------------------------------------------------------------------------------------------------------


    subroutine ft_read_keyword_header_logical(header, keyword, value, found, status, comment)

        character(len=*), intent(in)                       :: header
        character(len=*), intent(in)                       :: keyword
        logical, intent(out)                               :: value
        logical, intent(out), optional                     :: found
        integer, intent(out)                               :: status
        character(len=FLEN_COMMENT), optional, intent(out) :: comment

        character(len=FLEN_VALUE) :: charvalue
        logical                   :: found_

        value = .false.

        call get_keyword(header, keyword, charvalue, found_, comment, .not. present(found), status)
        if (present(found)) found = found_
        if (status /= 0 .or. .not. found_) return

        charvalue = strupcase(charvalue)
        if (charvalue /= 'FALSE'(1:min(5,len_trim(charvalue))) .and. charvalue /= 'TRUE'(1:min(4,len_trim(charvalue)))) then
            status = 404
            write (ERROR_UNIT,'(a)') "ft_read_keyword_header_logical: invalid logical value '" // trim(charvalue) //             &
                  "' for keyword '" // keyword // "' in FITS header."
            return
        end if

        if (charvalue(1:1) == 'T') value = .true.

    end subroutine ft_read_keyword_header_logical


    !-------------------------------------------------------------------------------------------------------------------------------


    subroutine ft_read_keyword_header_int4(header, keyword, value, found, status, comment)

        character(len=*), intent(in)                       :: header
        character(len=*), intent(in)                       :: keyword
        integer*4, intent(out)                             :: value
        logical, intent(out), optional                     :: found
        integer, intent(out)                               :: status
        character(len=FLEN_COMMENT), optional, intent(out) :: comment

        character(len=FLEN_VALUE) :: charvalue
        logical                   :: found_

        value = 0

        call get_keyword(header, keyword, charvalue, found_, comment, .not. present(found), status)
        if (present(found)) found = found_
        if (status /= 0 .or. .not. found_) return

        read (charvalue,'(i20)',iostat=status) value
        if (status /= 0) then
            status = 407
            write (ERROR_UNIT,'(a)') "ft_read_keyword_header_int4: invalid integer value '" // trim(charvalue) //                &
                  "' for keyword '" // keyword // "' in FITS header."
            return
        end if

    end subroutine ft_read_keyword_header_int4


    !-------------------------------------------------------------------------------------------------------------------------------


    subroutine ft_read_keyword_header_int8(header, keyword, value, found, status, comment)

        character(len=*), intent(in)                       :: header
        character(len=*), intent(in)                       :: keyword
        integer*8, intent(out)                             :: value
        logical, intent(out), optional                     :: found
        integer, intent(out)                               :: status
        character(len=FLEN_COMMENT), optional, intent(out) :: comment

        character(len=FLEN_VALUE) :: charvalue
        logical                   :: found_

        value = 0

        call get_keyword(header, keyword, charvalue, found_, comment, .not. present(found), status)
        if (present(found)) found = found_
        if (status /= 0 .or. .not. found_) return

        read (charvalue,'(i20)',iostat=status) value
        if (status /= 0) then
            status = 407
            write (ERROR_UNIT,'(a)') "ft_read_keyword_header_int8: invalid integer value '" // trim(charvalue) //                &
                  "' for keyword '" // keyword // "' in FITS header."
            return
        end if

    end subroutine ft_read_keyword_header_int8


    !-------------------------------------------------------------------------------------------------------------------------------


    subroutine ft_read_keyword_header_real4(header, keyword, value, found, status, comment)

        character(len=*), intent(in)                       :: header
        character(len=*), intent(in)                       :: keyword
        real(sp), intent(out)                              :: value
        logical, intent(out), optional                     :: found
        integer, intent(out)                               :: status
        character(len=FLEN_COMMENT), optional, intent(out) :: comment

        real(dp) :: value_

        call ft_read_keyword(header, keyword, value_, found, status, comment)
        value = real(value_, sp)

    end subroutine ft_read_keyword_header_real4


    !-------------------------------------------------------------------------------------------------------------------------------


    subroutine ft_read_keyword_header_real8(header, keyword, value, found, status, comment)

        character(len=*), intent(in)                       :: header
        character(len=*), intent(in)                       :: keyword
        real(dp), intent(out)                              :: value
        logical, intent(out), optional                     :: found
        integer, intent(out)                               :: status
        character(len=FLEN_COMMENT), optional, intent(out) :: comment

        character(len=FLEN_VALUE) :: charvalue
        logical                   :: found_

        value = 0

        call get_keyword(header, keyword, charvalue, found_, comment, .not. present(found), status)
        if (present(found)) found = found_
        if (status /= 0 .or. .not. found_) return

        read (charvalue,*,iostat=status) value
        if (status /= 0) then
            status = 409
            write (ERROR_UNIT,'(a)') "ft_read_keyword: invalid real value '" // trim(charvalue) //                 &
                  "' for keyword '" // keyword // "' in FITS header."
            return
        end if

    end subroutine ft_read_keyword_header_real8


    !-------------------------------------------------------------------------------------------------------------------------------


#if PRECISION_REAL == 16
    subroutine ft_read_keyword_header_real16(header, keyword, value, found, status, comment)

        character(len=*), intent(in)                       :: header
        character(len=*), intent(in)                       :: keyword
        real(qp), intent(out)                              :: value
        logical, intent(out), optional                     :: found
        integer, intent(out)                               :: status
        character(len=FLEN_COMMENT), optional, intent(out) :: comment

        real(dp) :: value_

        call ft_read_keyword(header, keyword, value_, found, status, comment)
        value = real(value_, qp)

    end subroutine ft_read_keyword_header_real16
#endif

    !-------------------------------------------------------------------------------------------------------------------------------


    subroutine ft_read_keyword_header_character(header, keyword, value, found, status, comment)

        character(len=*), intent(in)                       :: header
        character(len=*), intent(in)                       :: keyword
        character(len=*), intent(out)                      :: value
        logical, intent(out), optional                     :: found
        integer, intent(out)                               :: status
        character(len=FLEN_COMMENT), optional, intent(out) :: comment

        character(len=FLEN_VALUE) :: charvalue
        integer                   :: ncharvalue, i, j
        logical                   :: found_

        call get_keyword(header, keyword, charvalue, found_, comment, .not. present(found), status)
        if (present(found)) found = found_
        if (status /= 0 .or. .not. found_) return

        if (charvalue(1:1) /= "'") then
           value = charvalue
           return
        end if

        value = ' '

        ! remove leading quote
        charvalue = charvalue(2:len_trim(charvalue))

        ! scan until next '
        ! make double quotes single
        ncharvalue = len_trim(charvalue)
        i = 1
        j = 1
        do while (j <= ncharvalue)
           if (charvalue(j:j) == "'") then
              if (j == ncharvalue .or. charvalue(j+1:j+1) /= "'") exit
           end if
           value(i:i) = charvalue(j:j)
           i = i + 1
           if (j /= ncharvalue) then
               if (charvalue(j:j+1) == "''") j = j + 1
           end if
           j = j + 1
        end do

    end subroutine ft_read_keyword_header_character


end module module_fitstools
