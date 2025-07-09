###########################
# MIT License
#
# Copyright (c) 2025 Nicholas Huebner Julian njulian@ucla.edu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
############################
import numpy as np
import torch
class SparseBins:
    def __init__( self, data:np.ndarray, minBinWidth, maxBinWidth):
        self.data = data
        if not isinstance( self.data, np.ndarray):
            if isinstance( self.data, torch.Tensor):
                self.dat = self.data.numpy()
            else:
                self.data = np.array(self.data)
        if maxBinWidth < minBinWidth:
            raise RuntimeError( f"maxBinWidth ({maxBinWidth}) < minBinWidth ({minBinWidth})")
        if np.max( data) > 1 or np.min( data) < 0:
            raise RuntimeError( f"data passed to SparseBins must be normalized to the unit hypercube, but found np.min( data) {np.min(data)} and np.max(data) {np.max(data)}")
        self.minBinWidth = minBinWidth
        self.maxBinWidth = maxBinWidth

        self.binCenters = (0.5*np.ones((1,self.data.shape[1]))).tolist()
        self.binHalfWidths = [0.5]
        self.binContents = [np.arange( self.data.shape[0])] # Indices of contents. Initial bin contains all
        self._indicesOfBinsToPop = []

        # construct a matrix which determines signs of difference vectors
        self._dim = self.data.shape[1]
        self._negate = np.expand_dims( np.arange(2**self._dim), -1).view( np.uint8)
        if self._negate.dtype.byteorder == '>' or (self._negate.dtype.byteorder == '=' and sys.byteorder == 'big'):
            self._negate = self._negate[:,::-1] # enforce little endian
        self._negate = np.unpackbits( self._negate, axis=1, count=self._dim, bitorder='little')
        self._negate = np.ones([2**self._dim, self._dim], dtype=int) - 2*self._negate[:,::-1]

        # reduce bins to satisfy maxBinWidth
        self.split_occupied_bins()

    def _recursively_split_a_bin( self, idx):
        if 2*self.binHalfWidths[ idx] > self.maxBinWidth \
            or ( len( self.binContents[ idx]) > 1 \
                and 2*self.binHalfWidths[ idx] > self.minBinWidth
                ):

                self._indicesOfBinsToPop.append( idx)
                prevCount = len(self.binHalfWidths)
                newCount = prevCount

                # create sub-bins
                newBinHalfWidth = 0.5 * self.binHalfWidths[ idx]
                newCenters = self.binCenters[ idx] + np.ones( self._negate.shape) * newBinHalfWidth * self._negate

                dataSlice = self.data[ self.binContents[ idx], :]
                for center in newCenters:
                    maskLower = ( dataSlice >= np.array( center) - newBinHalfWidth).all( axis=1)

                    # accomodate the case when data equals the upper bound of the entire system
                    upperBounds = np.array( center) + newBinHalfWidth
                    upperBounds[ upperBounds == 1.0] = 1.0 + np.finfo( upperBounds.dtype).resolution
                    maskUpper = ( dataSlice < upperBounds).all( axis=1)
                    mask = maskLower * maskUpper

                    # retrieve the indices of elements of mask which are True
                    if mask.any():
                        self.binContents.append( np.array( self.binContents[ idx])[ mask].tolist())
                        self.binHalfWidths.append( newBinHalfWidth)
                        self.binCenters.append( center.tolist())
                        newCount += 1
                for newBinIdx in range( prevCount, newCount):
                    self._recursively_split_a_bin( newBinIdx)
        return

    def split_occupied_bins( self):
        self._indicesOfBinsToPop = []
        # iterate over bins and split them if appropriate
        for oldBinIdx in range( len( self.binHalfWidths)):
            self._recursively_split_a_bin( oldBinIdx)

        for ii in reversed( sorted( self._indicesOfBinsToPop)):
            self.binHalfWidths.pop( ii)
            self.binCenters.pop( ii)
            self.binContents.pop( ii)
        self._indicesOfBinsToPop.clear()
        count = 0
        for ii in range( len( self.binCenters)):
            count += len( self.binContents[ ii])
        if count != self.data.shape[0]:
            print(f"count {count}, self.data.shape[0] {self.data.shape[0]}")
            raise Exception(f"data lost: count != self.data.shape[0]")
        return self

    def volumes( self):
        volumes = []
        for ii in range( len( self.binContents)):
            volumes.append( (2*self.binHalfWidths[ii])**(self._dim))
        return volumes

    def show( self):
        print(f"max( self.binHalfWidths): {np.max( self.binHalfWidths)}")
        print(f"min( self.binHalfWidths): {np.min( self.binHalfWidths)}")
        print(f"(minBinWidth, maxBinWidth): ({self.minBinWidth}, {self.maxBinWidth})")
        print(f"total volume of bins: {np.sum( self.volumes())}")
