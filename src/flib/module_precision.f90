! Copyright 2010-2011 Pierre Chanial
! All rights reserved
!
module module_precision

   integer, parameter :: sp = kind(1.0)
   integer, parameter :: dp = selected_real_kind(2*precision(1.0_sp))
   integer, parameter :: qp = selected_real_kind(2*precision(1.0_dp))
   
end module module_precision
