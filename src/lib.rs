#![feature(stdarch_nvptx)]
#![no_std]

use core::arch::nvptx::*;
use core::marker::PhantomData;
use ndarray::prelude::*;

trait Space {
    fn index_x() -> usize;
    fn index_y() -> usize;
    fn index_z() -> usize;
    fn size_x() -> usize;
    fn size_y() -> usize;
    fn size_z() -> usize;
}

struct BlockSpace;
impl Space for BlockSpace {
    fn index_x() -> usize {
        unsafe { _thread_idx_x() as usize }
    }

    fn index_y() -> usize {
        unsafe { _thread_idx_y() as usize }
    }

    fn index_z() -> usize {
        unsafe { _thread_idx_z() as usize }
    }

    fn size_x() -> usize {
        unsafe { _block_dim_x() as usize }
    }

    fn size_y() -> usize {
        unsafe { _block_dim_y() as usize }
    }

    fn size_z() -> usize {
        unsafe { _block_dim_z() as usize }
    }
}

struct GlobalSpace;
impl Space for GlobalSpace {
    fn index_x() -> usize {
        unsafe { (_thread_idx_x() + _block_dim_x() * _block_idx_x()) as usize }
    }

    fn index_y() -> usize {
        unsafe { (_thread_idx_y() + _block_dim_y() * _block_idx_y()) as usize }
    }

    fn index_z() -> usize {
        unsafe { (_thread_idx_z() + _block_dim_z() * _block_idx_z()) as usize }
    }

    fn size_x() -> usize {
        unsafe { (_block_dim_x() * _grid_dim_x()) as usize }
    }

    fn size_y() -> usize {
        unsafe { (_block_dim_y() * _grid_dim_y()) as usize }
    }

    fn size_z() -> usize {
        unsafe { (_block_dim_z() * _grid_dim_z()) as usize }
    }
}

trait IndexSource {
    fn produce_index() -> usize;
}

struct X<SPACE>(PhantomData<SPACE>);
impl<SPACE: Space> IndexSource for X<SPACE> {
    fn produce_index() -> usize {
        SPACE::index_x()
    }
}
struct Y<SPACE>(PhantomData<SPACE>);
impl<SPACE: Space> IndexSource for Y<SPACE> {
    fn produce_index() -> usize {
        SPACE::index_y()
    }
}
struct Z<SPACE>(PhantomData<SPACE>);
impl<SPACE: Space> IndexSource for Z<SPACE> {
    fn produce_index() -> usize {
        SPACE::index_z()
    }
}
struct XY<SPACE>(PhantomData<SPACE>);
impl<SPACE: Space> IndexSource for XY<SPACE> {
    fn produce_index() -> usize {
        SPACE::index_x() + SPACE::index_y() * SPACE::size_x()
    }
}
struct XZ<SPACE>(PhantomData<SPACE>);
impl<SPACE: Space> IndexSource for XZ<SPACE> {
    fn produce_index() -> usize {
        SPACE::index_x() + SPACE::index_z() * SPACE::size_x()
    }
}
struct YZ<SPACE>(PhantomData<SPACE>);
impl<SPACE: Space> IndexSource for YZ<SPACE> {
    fn produce_index() -> usize {
        SPACE::index_y() + SPACE::index_z() * SPACE::size_y()
    }
}
struct XYZ<SPACE>(PhantomData<SPACE>);
impl<SPACE: Space> IndexSource for XYZ<SPACE> {
    fn produce_index() -> usize {
        SPACE::index_x()
            + SPACE::index_y() * SPACE::size_x()
            + SPACE::index_z() * SPACE::size_x() * SPACE::size_y()
    }
}

pub struct SimtArr<A, D, P> {
    inner: RawArrayViewMut<A, D>,

    xyz_marker: PhantomData<P>,
}

trait MissingX {
    type Target;
    fn assoc_x(self) -> Self::Target;
}

trait MissingY {
    type Target;
    fn assoc_y(self) -> Self::Target;
}

trait MissingZ {
    type Target;
    fn assoc_z(self) -> Self::Target;
}
trait MissingXY: MissingX + MissingY {}
trait MissingXZ: MissingX + MissingZ {}
trait MissingYZ: MissingY + MissingZ {}
trait MissingXYZ: MissingX + MissingY + MissingZ {}

struct MissingXType;
