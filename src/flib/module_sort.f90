! Copyright 2010-2011 Pierre Chanial
! All rights reserved
!
module module_sort

    use module_math_old,    only : NaN, neq_real
    use module_tamasis, only : p
    implicit none
    private

    public :: histogram
    public :: reorder
    public :: qsortgi
    public :: qsorti
    public :: qsortid
    public :: uniq
    public :: where

    integer, pointer   :: module_array_int(:) => null()
    real(p), pointer   :: module_array_double(:) => null()
    logical*1, pointer :: module_mask(:) => null()
    !$omp threadprivate(module_array_int, module_array_double, module_mask)

    interface histogram
        module procedure histogram_nomask_int, histogram_mask_int
    end interface histogram

    interface reorder
        module procedure reorder_nomask_double, reorder_mask_double
    end interface reorder

    interface qsorti
        module procedure qsorti_int, qsorti_nomask_double, qsorti_mask_double
    end interface qsorti

    interface uniq
        module procedure uniq_int, uniq_double
    end interface uniq

    interface where
        module procedure where_1d_1d, where_2d_1d, where_2d_2d, where_3d_3d
    end interface where

contains

! From HDK@psuvm.psu.edu Thu Dec  8 15:27:16 MST 1994
!
! The following was converted from Algol recursive to Fortran iterative
! by a colleague at Penn State (a long time ago - Fortran 66, please
! excuse the GoTo's). The following code also corrects a bug in the
! Quicksort algorithm published in the ACM (see Algorithm 402, CACM,
! Sept. 1970, pp 563-567; also you younger folks who weren't born at
! that time might find interesting the history of the Quicksort
! algorithm beginning with the original published in CACM, July 1961,
! pp 321-322, Algorithm 64). Note that the following algorithm sorts
! integer data; actual data is not moved but sort is affected by sorting
! a companion index array (see leading comments). The data type being
! sorted can be changed by changing one line; see comments after
! declarations and subsequent one regarding comparisons(Fortran
! 77 takes care of character comparisons of course, so that comment
! is merely historical from the days when we had to write character
! compare subprograms, usually in assembler language for a specific
! mainframe platform at that time). But the following algorithm is
! good, still one of the best available.


      SUBROUTINE QSORTID (A, ORD)
      implicit none
      integer :: i
      integer :: ip
      integer :: iq
      integer :: ix
      integer :: iz
      integer :: l
      integer :: l1
      integer :: ndeep
      integer :: p_
      integer :: q
      integer :: u
      integer :: u1
      integer :: yp
!
      real(p), intent(in)  :: A(:)
      integer, intent(out) :: ORD(size(A))
      integer :: POPLST(2,1000)
      integer :: N
      real(p) :: X,XX,Z,ZZ,Y
!
      n = size(A)
!     TO SORT DIFFERENT INPUT TYPES, CHANGE THE FOLLOWING
!     SPECIFICATION STATEMENTS; FOR EXAMPLE, FOR FORTRAN CHARACTER
!     USE THE FOLLOWING:  CHARACTER *(*) A(N)
!
      NDEEP=0
      U1=N
      L1=1
      DO 1  I=1,N
    1 ORD(I)=I
    2 IF (U1.LE.L1) RETURN
!
    3 L=L1
      U=U1
!
! PART
!
    4 P_=L
      Q=U
!     FOR CHARACTER SORTS, THE FOLLOWING 3 STATEMENTS WOULD BECOME
!     X = ORD(P_)
!     Z = ORD(Q)
!     IF (A(X) .LE. A(Z)) GO TO 2
!
!     WHERE "CLE" IS A LOGICAL FUNCTION WHICH RETURNS "TRUE" IF THE
!     FIRST ARGUMENT IS LESS THAN OR EQUAL TO THE SECOND, BASED ON "LEN"
!     CHARACTERS.
!
      X=A(ORD(P_))
      Z=A(ORD(Q))
      IF (X.LE.Z) GO TO 5
      Y=X
      X=Z
      Z=Y
      YP=ORD(P_)
      ORD(P_)=ORD(Q)
      ORD(Q)=YP
    5 IF (U-L.LE.1) GO TO 15
      XX=X
      IX=P_
      ZZ=Z
      IZ=Q
!
! LEFT
!
    6 P_=P_+1
      IF (P_.GE.Q) GO TO 7
      X=A(ORD(P_))
      IF (X.GE.XX) GO TO 8
      GO TO 6
    7 P_=Q-1
      GO TO 13
!
! RIGHT
!
    8 Q=Q-1
      IF (Q.LE.P_) GO TO 9
      Z=A(ORD(Q))
      IF (Z.LE.ZZ) GO TO 10
      GO TO 8
    9 Q=P_
      P_=P_-1
      Z=X
      X=A(ORD(P_))
!
! DIST
!
   10 IF (X.LE.Z) GO TO 11
      Y=X
      X=Z
      Z=Y
      IP=ORD(P_)
      ORD(P_)=ORD(Q)
      ORD(Q)=IP
   11 IF (X.LE.XX) GO TO 12
      XX=X
      IX=P_
   12 IF (Z.GE.ZZ) GO TO 6
      ZZ=Z
      IZ=Q
      GO TO 6
!
! OUT
!
   13 CONTINUE
      IF (.NOT.(P_.NE.IX.AND.X.NE.XX)) GO TO 14
      IP=ORD(P_)
      ORD(P_)=ORD(IX)
      ORD(IX)=IP
   14 CONTINUE
      IF (.NOT.(Q.NE.IZ.AND.Z.NE.ZZ)) GO TO 15
      IQ=ORD(Q)
      ORD(Q)=ORD(IZ)
      ORD(IZ)=IQ
   15 CONTINUE
      IF (U-Q.LE.P_-L) GO TO 16
      L1=L
      U1=P_-1
      L=Q+1
      GO TO 17
   16 U1=U
      L1=Q+1
      U=P_-1
   17 CONTINUE
      IF (U1.LE.L1) GO TO 18
!
! START RECURSIVE CALL
!
      NDEEP=NDEEP+1
      POPLST(1,NDEEP)=U
      POPLST(2,NDEEP)=L
      GO TO 3
   18 IF (U.GT.L) GO TO 4
!
! POP BACK UP IN THE RECURSION LIST
!
      IF (NDEEP.EQ.0) GO TO 2
      U=POPLST(1,NDEEP)
      L=POPLST(2,NDEEP)
      NDEEP=NDEEP-1
      GO TO 18
!
! END SORT
! END QSORT
!
      end subroutine QSORTID

      SUBROUTINE QSORTGI (nitems, compare, ORD)
      implicit none
      integer, intent(in) :: nitems
      integer, intent(out) :: ORD(nitems)
      interface
          integer function compare(first, second)
              integer, intent(in) :: first, second
          end function compare
      end interface

      integer :: i
      integer :: ip
      integer :: iq
      integer :: ix
      integer :: iz
      integer :: l
      integer :: l1
      integer :: ndeep
      integer :: p_
      integer :: q
      integer :: u
      integer :: u1
      integer :: yp
!
      integer :: POPLST(2,1000)
      integer :: X,XX,Z,ZZ,Y
!
!     TO SORT DIFFERENT INPUT TYPES, CHANGE THE FOLLOWING
!     SPECIFICATION STATEMENTS; FOR EXAMPLE, FOR FORTRAN CHARACTER
!     USE THE FOLLOWING:  CHARACTER *(*) A(N)
!
      NDEEP=0
      U1=NITEMS
      L1=1
      DO I=1,NITEMS
          ORD(I)=I
      END DO

    2 IF (U1.LE.L1) RETURN
!
    3 L=L1
      U=U1
!
! PART
!
    4 P_=L
      Q=U
!     FOR CHARACTER SORTS, THE FOLLOWING 3 STATEMENTS WOULD BECOME
!     X = ORD(P_)
!     Z = ORD(Q)
!     IF (A(X) .LE. A(Z)) GO TO 2
!
!     WHERE "CLE" IS A LOGICAL FUNCTION WHICH RETURNS "TRUE" IF THE
!     FIRST ARGUMENT IS LESS THAN OR EQUAL TO THE SECOND, BASED ON "LEN"
!     CHARACTERS.
!
      X=ORD(P_)
      Z=ORD(Q)
      IF (compare(X, Z) >= 0) GO TO 5
      Y = X
      X = Z
      Z = Y
      YP=ORD(P_)
      ORD(P_)=ORD(Q)
      ORD(Q)=YP
    5 IF (U - L <= 1) GO TO 15
      XX=X
      IX=P_
      ZZ=Z
      IZ=Q
!
! LEFT
!
    6 P_=P_+1
      IF (P_.GE.Q) GO TO 7
      X=ORD(P_)
      IF (compare(X, XX) <= 0) GO TO 8
      GO TO 6
    7 P_=Q-1
      GO TO 13
!
! RIGHT
!
    8 Q=Q-1
      IF (Q.LE.P_) GO TO 9
      Z=ORD(Q)
      IF (compare(Z, ZZ) >= 0) GO TO 10
      GO TO 8
    9 Q=P_
      P_=P_-1
      Z=X
      X=ORD(P_)
!
! DIST
!
   10 IF (compare(X, Z) >= 0) GO TO 11
      Y = X
      X = Z
      Z = Y
      IP=ORD(P_)
      ORD(P_)=ORD(Q)
      ORD(Q)=IP
   11 IF (compare(X, XX) >= 0) GO TO 12
      XX=X
      IX=P_
   12 IF (compare(Z, ZZ) <=0) GO TO 6
      ZZ=Z
      IZ=Q
      GO TO 6
!
! OUT
!
   13 CONTINUE
      IF (.NOT.(P_ /= IX .AND. compare(X,XX) /= 0)) GO TO 14
      IP=ORD(P_)
      ORD(P_)=ORD(IX)
      ORD(IX)=IP
   14 CONTINUE
      IF (.NOT.(Q /= IZ .AND. compare(Z,ZZ) /= 0)) GO TO 15
      IQ=ORD(Q)
      ORD(Q)=ORD(IZ)
      ORD(IZ)=IQ
   15 CONTINUE
      IF (U-Q <= P_-L) GO TO 16
      L1=L
      U1=P_-1
      L=Q+1
      GO TO 17
   16 U1=U
      L1=Q+1
      U=P_-1
   17 CONTINUE
      IF (U1 <= L1) GO TO 18
!
! START RECURSIVE CALL
!
      NDEEP=NDEEP+1
      POPLST(1,NDEEP)=U
      POPLST(2,NDEEP)=L
      GO TO 3
   18 IF (U.GT.L) GO TO 4
!
! POP BACK UP IN THE RECURSION LIST
!
      IF (NDEEP.EQ.0) GO TO 2
      U=POPLST(1,NDEEP)
      L=POPLST(2,NDEEP)
      NDEEP=NDEEP-1
      GO TO 18
!
! END SORT
! END QSORT
!
    end subroutine qsortgi


    !-------------------------------------------------------------------------------------------------------------------------------
    ! INTEGER
    !-------------------------------------------------------------------------------------------------------------------------------

    subroutine qsorti_int(array, index)

        integer, intent(in), target :: array(:)
        integer, intent(out) :: index(size(array))

        module_array_int => array
        call qsortgi(size(array), compare_int, index)
        module_array_int => null()

    end subroutine qsorti_int


    !-------------------------------------------------------------------------------------------------------------------------------


    function compare_int(first, second)

        integer, intent(in) :: first, second
        integer             :: compare_int

        if (module_array_int(first) < module_array_int(second)) then
           compare_int = 1
        else if (module_array_int(first) /= module_array_int(second)) then
           compare_int = -1
        else
           compare_int = 0
        end if

    end function compare_int


    !-------------------------------------------------------------------------------------------------------------------------------
    ! DOUBLE
    !-------------------------------------------------------------------------------------------------------------------------------

    subroutine qsorti_nomask_double(array, index)

        real(p), intent(in), target :: array(:)
        integer, intent(out)        :: index(size(array))

        module_array_double => array
        call qsortgi(size(array), compare_nomask_double, index)
        module_array_double => null()

    end subroutine qsorti_nomask_double


    !-------------------------------------------------------------------------------------------------------------------------------


    subroutine qsorti_mask_double(array, mask, index)

        real(p), intent(in), target   :: array(:)
        logical*1, intent(in), target :: mask(:)
        integer, intent(out)          :: index(size(array))

        module_array_double => array
        module_mask => mask
        call qsortgi(size(array), compare_mask_double, index)
        module_array_double => null()
        module_mask => null()

    end subroutine qsorti_mask_double


    !-------------------------------------------------------------------------------------------------------------------------------


    function compare_nomask_double(first, second) result (compare)

        integer, intent(in) :: first, second
        integer             :: compare

        if (module_array_double(first) /= module_array_double(first)) then
            if (module_array_double(second) /= module_array_double(second)) then
                compare = 0
            else
                compare = -1
            end if
        else if (module_array_double(second) /= module_array_double(second)) then
            compare = 1
        else if (module_array_double(first) < module_array_double(second)) then
           compare = 1
        else if (module_array_double(first) /= module_array_double(second)) then
           compare = -1
        else
           compare = 0
        end if

    end function compare_nomask_double


    !-------------------------------------------------------------------------------------------------------------------------------


    function compare_mask_double(first, second) result (compare)

        integer, intent(in) :: first, second
        integer             :: compare

        if (module_array_double(first) /= module_array_double(first) .or. module_mask(first)) then
            if (module_array_double(second) /= module_array_double(second) .or. module_mask(second)) then
                compare = 0
            else
                compare = -1
            end if
        else if (module_array_double(second) /= module_array_double(second) .or. module_mask(second)) then
            compare = 1
        else if (module_array_double(first) < module_array_double(second)) then
            compare = 1
        else if (module_array_double(first) /= module_array_double(second)) then
            compare = -1
        else
            compare = 0
        end if

    end function compare_mask_double


    !-------------------------------------------------------------------------------------------------------------------------------


    subroutine uniq_int(array, outindex)

        integer, intent(in)  :: array(:)
        integer, intent(out), allocatable :: outindex(:)

        integer :: i, count, n, val
#if defined(GFORTRAN) || defined(IFORT)
        integer, allocatable :: tmp(:)
#endif

        n = size(array)
        if (n == 0) then
            allocate (outindex(0))
            return
        end if

        allocate (outindex(n))
        count = 0
        val = array(1)
        do i = 2, n
            if (array(i) /= val) then
                count = count + 1
                outindex(count) = i-1
                val = array(i)
            end if
        end do
        count = count + 1
        outindex(count) = n

#if defined(GFORTRAN) || defined(IFORT)
        allocate (tmp(count))
        tmp = outindex(1:count)
        call move_alloc(tmp, outindex)
#else
        outindex = outindex(1:count)
#endif

    end subroutine uniq_int


    !-------------------------------------------------------------------------------------------------------------------------------


    ! fixme: algo different from uniq_int...
    subroutine uniq_double(array, inindex, outindex, rtol)

        real(p), intent(in)               :: array(:)
        integer, intent(in)               :: inindex(size(array))
        integer, intent(out), allocatable :: outindex(:)
        real(p), intent(in)               :: rtol

        integer :: index(size(array)), nuniqs, n, i
        real(p) :: val

        n = size(array)
        if (n == 0) then
            allocate (outindex(0))
            return
        end if

        nuniqs = 1
        index(1) = inindex(1)
        val = array(inindex(1))
        do i = 2, n
            if (array(inindex(i)) /= array(inindex(i))) cycle
            if (neq_real(array(inindex(i)), val, rtol)) then
                val = array(inindex(i))
                nuniqs = nuniqs + 1
                index(nuniqs) = inindex(i)
            end if
        end do
        
        allocate (outindex(nuniqs))
        outindex = index(1:nuniqs)

    end subroutine uniq_double


    !-------------------------------------------------------------------------------------------------------------------------------


    subroutine reorder_nomask_double(array, index, nuniqs, table, rtol)

        use iso_fortran_env, only : OUTPUT_UNIT
        implicit none

        real(p), intent(in)               :: array(:)
        integer, intent(out)              :: index(size(array))
        integer, intent(out)              :: nuniqs
        real(p), intent(out), allocatable :: table(:)
        real(p), intent(in)               :: rtol

        integer :: isort(size(array)), ndata, i
        real(p) :: val, val2, tmptable(size(array))

        ndata = size(array)
        if (ndata == 0) then
            nuniqs = 0
            allocate (table(0))
            return
        end if

        call qsorti(array, isort)

        nuniqs = 0
        val = NaN
        do i = 1, ndata
            val2 = array(isort(i))
            if (val2 /= val2) then
                index(isort(i)) = huge(index)
                cycle
            end if
            if (neq_real(val, val2, rtol)) then
                val = val2
                nuniqs = nuniqs + 1
                tmptable(nuniqs) = val
            end if
            index(isort(i)) = nuniqs
        end do

        allocate (table(nuniqs))
        table = tmptable(:nuniqs)
        
    end subroutine reorder_nomask_double


    !-------------------------------------------------------------------------------------------------------------------------------


    subroutine reorder_mask_double(array, mask, index, nuniqs, table, rtol)

        use iso_fortran_env, only : OUTPUT_UNIT
        implicit none

        real(p), intent(in)               :: array(:)
        logical*1, intent(in)             :: mask(size(array))
        integer, intent(out)              :: index(size(array))
        integer, intent(out)              :: nuniqs
        real(p), intent(out), allocatable :: table(:)
        real(p), intent(in)               :: rtol

        integer :: isort(size(array)), ndata, i
        real(p) :: val, val2, tmptable(size(array))

        ndata = size(array)
        if (ndata == 0) then
            nuniqs = 0
            allocate (table(0))
            return
        end if

        call qsorti(array, mask, isort)

        nuniqs = 0
        val = NaN
        do i = 1, ndata
            val2 = array(isort(i))
            if (val2 /= val2 .or. mask(isort(i))) then
                index(isort(i)) = huge(index)
                cycle
            end if
            if (neq_real(val, val2, rtol)) then
                val = val2
                nuniqs = nuniqs + 1
                tmptable(nuniqs) = val
            end if
            index(isort(i)) = nuniqs
        end do

        allocate (table(nuniqs))
        table = tmptable(:nuniqs)
        
    end subroutine reorder_mask_double


    !-------------------------------------------------------------------------------------------------------------------------------


    function histogram_nomask_int(array, nbins) result(histogram)

        integer, intent(in) :: array(:)
        integer, intent(in) :: nbins
        integer             :: histogram(nbins)

        integer             :: i

        histogram = 0
        do i = 1, size(array)
            if (array(i) == huge(array)) cycle
            histogram(array(i)) = histogram(array(i)) + 1
        end do
        
    end function histogram_nomask_int


    !-------------------------------------------------------------------------------------------------------------------------------


    function histogram_mask_int(array, mask, nbins) result(histogram)

        integer, intent(in)   :: array(:)
        logical*1, intent(in) :: mask(size(array))
        integer, intent(in)   :: nbins
        integer               :: histogram(nbins)

        integer               :: i

        histogram = 0
        do i = 1, size(array)
            if (array(i) == huge(array) .or. mask(i)) cycle
            histogram(array(i)) = histogram(array(i)) + 1
        end do
        
    end function histogram_mask_int


    !-------------------------------------------------------------------------------------------------------------------------------


    subroutine where_1d_1d(array, indices, count)
        
        logical, intent(in) :: array(:)
        integer, intent(out), allocatable :: indices(:)
        integer, intent(out), optional :: count

        integer :: indices_(size(array))
        integer :: i, count_

        count_ = 0
        do i = 1, size(array)
            if (array(i)) then
                count_ = count_ + 1
                indices_(count_) = i
            end if
        end do
        
        allocate (indices(count_))
        indices = indices_(1:count_)
        if (present(count)) count = count_
        
    end subroutine where_1d_1d


    !-------------------------------------------------------------------------------------------------------------------------------


    subroutine where_2d_1d(mask, indices, n)
        
        logical, intent(in) :: mask(:,:)
        integer, intent(out), allocatable :: indices(:)
        integer, intent(out), optional :: n

        integer :: i

#if defined(GFORTRAN) || defined(IFORT)
        allocate (indices(count(mask)))
#endif
        indices = pack([(i, i=1, size(mask))], reshape(mask,[size(mask)]))
        n = size(indices)
        
    end subroutine where_2d_1d


    !-------------------------------------------------------------------------------------------------------------------------------


    subroutine where_2d_2d(array, i1, i2, count)
        
        logical, intent(in) :: array(:,:)
        integer, intent(out), allocatable :: i1(:), i2(:)
        integer, intent(out), optional :: count

        integer :: i1_(size(array)), i2_(size(array))
        integer :: i, j, count_

        count_ = 0
        do j = 1, size(array,2)
            do i = 1, size(array,1)
                if (array(i,j)) then
                    count_ = count_ + 1
                    i1_(count_) = i
                    i2_(count_) = j
                end if
            end do
        end do
        
        allocate (i1(count_))
        allocate (i2(count_))
        i1 = i1_(1:count_)
        i2 = i2_(1:count_)
        if (present(count)) count = count_
        
    end subroutine where_2d_2d


    !-------------------------------------------------------------------------------------------------------------------------------


    subroutine where_3d_3d(array, i1, i2, i3, count)
        
        logical, intent(in) :: array(:,:,:)
        integer, intent(out), allocatable :: i1(:), i2(:), i3(:)
        integer, intent(out), optional :: count

        integer :: i1_(size(array)), i2_(size(array)), i3_(size(array))
        integer :: i, j, k, count_

        count_ = 0
        do k = 1, size(array,3)
            do j = 1, size(array,2)
                do i = 1, size(array,1)
                    if (array(i,j,k)) then
                        count_ = count_ + 1
                        i1_(count_) = i
                        i2_(count_) = j
                        i3_(count_) = k
                    end if
                end do
            end do
        end do

        allocate (i1(count_))
        allocate (i2(count_))
        allocate (i3(count_))
        i1 = i1_(1:count_)
        i2 = i2_(1:count_)
        i3 = i3_(1:count_)
        if (present(count)) count = count_
        
    end subroutine where_3d_3d


end module module_sort
