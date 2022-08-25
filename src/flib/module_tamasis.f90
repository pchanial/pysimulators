! Copyright 2010-2011 Pierre Chanial
! All rights reserved
!
module module_tamasis

    use iso_fortran_env,  only : OUTPUT_UNIT
    use module_precision, only : sp, dp, qp
    implicit none
    private

    public :: p, POLICY_KEEP, POLICY_MASK, POLICY_REMOVE
    public :: tamasis_dir  ! Tamasis data directory

    character(len=*), parameter :: tamasis_dir = ''
    integer, parameter :: p = dp

    integer, parameter :: POLICY_KEEP   = 0
    integer, parameter :: POLICY_MASK   = 1
    integer, parameter :: POLICY_REMOVE = 2

end module module_tamasis
