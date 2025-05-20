High-Level Programming Interface
================================

With APPy's high-level programming interface, all you need to do to parallelize a loop on GPUs 
is to replace ``range`` with ``appy.prange`` and annotate the loop with ``@appy.jit``. Here's an example: