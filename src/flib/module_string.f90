! Copyright 2010-2011 Pierre Chanial
! All rights reserved
!
module module_string

    use module_tamasis, only : p
    implicit none
    private

    public :: strlowcase
    public :: strupcase
    public :: strcompress
    public :: strsplit
    public :: strinteger
    public :: strjoin
    public :: strreal
    public :: strsection
    public :: strternary

    interface strjoin
        module procedure strjoin_trim, strjoin_opt
    end interface

    interface strinteger
        module procedure strinteger_int4, strinteger_int4_left, strinteger_int8
    end interface

    interface strsection
        module procedure strsection_int4, strsection_int8
    end interface


contains


    ! convert a word to lower case
    elemental function strlowcase(input)

        character(len=*), intent(in) :: input
        character(len=len(input))    :: strlowcase

        integer                      :: i,ic

        strlowcase = input
        do i=1, len(input)
            ic = iachar(input(i:i))
            if (ic >= 65 .and. ic < 90) strlowcase(i:i) = achar(ic+32)
        end do

     end function strlowcase


    !-------------------------------------------------------------------------------------------------------------------------------


    ! convert a word to upper case
    elemental function strupcase(input)

        character(len=*), intent(in) :: input
        character(len=len(input))    :: strupcase

        integer                      :: i,ic

        strupcase = input
        do i=1, len(input)
            ic = iachar(input(i:i))
            if (ic >= 97 .and. ic < 122) strupcase(i:i) = achar(ic-32)
        end do

    end function strupcase


    !-------------------------------------------------------------------------------------------------------------------------------


    ! remove blanks
    ! strcompress should have an allocatable length, but not implemented in gfortran 4.4
    elemental function strcompress(input)

        character(len=*), intent(in) :: input
        character(len=len(input))    :: strcompress

        integer                      :: i, j
  
        strcompress = ' '
        j = 1
        do i=1, len_trim(input)
            if (iachar(input(i:i)) == 32) cycle
            strcompress(j:j) = input(i:i)
            j = j + 1
        end do
      
    end function strcompress


    !-------------------------------------------------------------------------------------------------------------------------------


    subroutine strsplit(input, delimiter, output)

        character(len=*), intent(in) :: input
        character, intent(in)        :: delimiter
        character(len=*), allocatable, intent(out) :: output(:)

        integer :: i, noutputs, ioutput, start
      
        noutputs = 1
        do i = 1, len(input)
            if (input(i:i) == delimiter) then
                noutputs = noutputs + 1
            end if
        end do

        allocate (output(noutputs))

        ioutput = 1
        start = 1
        do i=1, len(input)
            if (input(i:i) == delimiter) then
                output(ioutput) = input(start:i-1)
                ioutput = ioutput + 1
                start = i + 1
                cycle
            end if
        end do
        output(ioutput) = input(start:)

    end subroutine strsplit


    !-------------------------------------------------------------------------------------------------------------------------------


    pure function strinteger_int4(input) result(strinteger)

        integer*4, intent(in)                     :: input
        character(len=strinteger_int4_len(input)) :: strinteger

        character(len=80)                         :: string
      
        string = ' '
        write(string, '(i80)') input
        strinteger = adjustl(string)
      
    end function strinteger_int4


    !-------------------------------------------------------------------------------------------------------------------------------


    pure function strinteger_int4_len(input) result(length)
 
        integer               :: length
        integer*4, intent(in) :: input

        if (input == 0)  then
            length = 1
            return
        end if

        length = floor(log10(dble(abs(input))))+1

        if (input < 0) length = length + 1

    end function strinteger_int4_len


    !-------------------------------------------------------------------------------------------------------------------------------


    pure function strinteger_int4_left(input, width) result(strinteger)

        integer*4, intent(in) :: input
        integer, intent(in)   :: width
        character(len=width)  :: strinteger

        write (strinteger, '(i0)') input
        strinteger = adjustl(strinteger)
      
    end function strinteger_int4_left


    !-------------------------------------------------------------------------------------------------------------------------------


    pure function strinteger_int8(input) result(strinteger)

        integer*8, intent(in)                     :: input
        character(len=strinteger_int8_len(input)) :: strinteger

        character(len=80)                         :: string

        string = ' '
        write(string, '(i80)') input
        strinteger = adjustl(string)
      
    end function strinteger_int8


    !-------------------------------------------------------------------------------------------------------------------------------


    pure function strinteger_int8_len(input) result(length)
 
        integer*8, intent(in) :: input
        integer               :: length

        if (input == 0)  then
            length = 1
            return
        end if

        length = floor(log10(dble(abs(input))))+1

        if (input < 0) length = length + 1

    end function strinteger_int8_len
 
 
    !-------------------------------------------------------------------------------------------------------------------------------


    pure function strinteger_int8_left(input, width) result(strinteger)

        integer*8, intent(in) :: input
        integer, intent(in)   :: width
        character(len=width)  :: strinteger

        write (strinteger, '(i0)') input
        strinteger = adjustl(strinteger)
      
    end function strinteger_int8_left


    !-------------------------------------------------------------------------------------------------------------------------------


    pure function strreal(input, prec)

        real(p), intent(in) :: input
        integer, intent(in) :: prec
        character(len=strreal_len(input,prec)) :: strreal
      
        character(len=20)   :: charvalue
        integer             :: status

        write (charvalue,'(bn,f20.'//strinteger(prec)//')',iostat=status) input
        if (status /= 0) then
            strreal = '*****'
            return
        end if
      
        strreal = adjustl(charvalue)

    end function strreal


    !-------------------------------------------------------------------------------------------------------------------------------


    pure function strreal_len(input, prec) result(length)

        integer             :: length
        real(p), intent(in) :: input
        integer, intent(in) :: prec

        character(len=20)   :: charvalue
        integer             :: status

        write (charvalue,'(bn,f20.'//strinteger(prec)//')',iostat=status) input
        if (status /= 0) then
            length = 5
            return
        end if
      
        length = len_trim(adjustl(charvalue))

    end function strreal_len


    !-------------------------------------------------------------------------------------------------------------------------------


    pure function strjoin_trim(input)

        character(len=*), intent(in)           :: input(:)
        character(len=strjoin_trim_len(input)) :: strjoin_trim

        integer                                :: i, k
        
        k = 1
        do i = 1, size(input)
            strjoin_trim(k:k+len_trim(input(i))-1) = trim(input(i))
            k = k + len_trim(input(i))
        end do

    end function strjoin_trim


    !-------------------------------------------------------------------------------------------------------------------------------


    pure function strjoin_trim_len(input)

        character(len=*), intent(in) :: input(:)
        integer                      :: strjoin_trim_len

        integer                      :: i

        strjoin_trim_len = sum([(len_trim(input(i)), i=1, size(input))])

    end function strjoin_trim_len


    !-------------------------------------------------------------------------------------------------------------------------------


    pure function strjoin_opt(input, dotrim)

        character(len=*), intent(in)                 :: input(:)
        logical, intent(in)                          :: dotrim
        character(len=strjoin_opt_len(input,dotrim)) :: strjoin_opt

        integer                                      :: i, k

        if (dotrim) then
            k = 1
            do i = 1, size(input)
                strjoin_opt(k:k+len_trim(input(i))-1) = trim(input(i))
                k = k + len_trim(input(i))
            end do
        else
            k = len(input)
            do i = 1, size(input)
                strjoin_opt((i-1)*k+1:i*k) = input(i)
            end do
        end if
        
    end function strjoin_opt


    !-------------------------------------------------------------------------------------------------------------------------------


    pure function strjoin_opt_len(input, dotrim)

        character(len=*), intent(in)  :: input(:)
        logical, intent(in)           :: dotrim
        integer                       :: strjoin_opt_len

        integer                       :: i

        if (.not. dotrim) then
            strjoin_opt_len = size(input) * len(input(1))
        else
            strjoin_opt_len = sum([(len_trim(input(i)), i=1, size(input))])
        end if

    end function strjoin_opt_len


    !-------------------------------------------------------------------------------------------------------------------------------


    function strsection_int4(first, last) result(str)

        integer*4, intent(in)                          :: first, last
        character(len=strsection_int4_len(first,last)) :: str
        
        if (first == 0 .and. last == 0) then
            str = ':'
            return
        end if

        if (first == 0) then
            write (str,'(":",i0)') last
            return
        end if

        if (last == 0) then
            write (str,'(i0,":")') first
            return
        end if

        write (str,'(i0,":",i0)') first, last

    end function strsection_int4


    !-------------------------------------------------------------------------------------------------------------------------------


    ! 0 means unbound
    pure function strsection_int4_len(first, last) result(length)

        integer               :: length
        integer*4, intent(in) :: first, last

        length = len(strinteger(first)) + len(strinteger(last)) + 1
        if (first == 0) length = length - 1
        if (last  == 0) length = length - 1

    end function strsection_int4_len


    !-------------------------------------------------------------------------------------------------------------------------------


    function strsection_int8(first, last) result(str)

        integer*8, intent(in)                          :: first, last
        character(len=strsection_int8_len(first,last)) :: str
        
        if (first == 0 .and. last == 0) then
            str = ':'
            return
        end if

        if (first == 0) then
            write (str,'(":",i0)') last
            return
        end if

        if (last == 0) then
            write (str,'(i0,":")') first
            return
        end if

        write (str,'(i0,":",i0)') first, last

    end function strsection_int8


    !-------------------------------------------------------------------------------------------------------------------------------


    ! 0 means unbound
    pure function strsection_int8_len(first, last) result(length)

        integer               :: length
        integer*8, intent(in) :: first, last

        length = len(strinteger(first)) + len(strinteger(last)) + 1
        if (first == 0) length = length - 1
        if (last  == 0) length = length - 1

    end function strsection_int8_len


    !-------------------------------------------------------------------------------------------------------------------------------


    function strternary(condition, str_true, str_false)

        logical, intent(in)          :: condition
        character(len=*), intent(in) :: str_true, str_false
        character(len=strternary_len(condition,str_true,str_false)):: strternary

        if (condition) then
            strternary = str_true
        else
            strternary = str_false
        end if

    end function strternary


    !-------------------------------------------------------------------------------------------------------------------------------


    pure function strternary_len(condition, str_true, str_false)

        integer                      :: strternary_len
        character(len=*), intent(in) :: str_true, str_false
        logical, intent(in)          :: condition

        if (condition) then
            strternary_len = len(str_true)
        else
            strternary_len = len(str_false)
        end if

    end function strternary_len


end module module_string
