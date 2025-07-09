#!/usr/bin/env python3
from SparseBins import SparseBins
import numpy as np
def main():
    minBinWidth=2**-5
    maxBinWidth=2**-3
    testData = np.vstack(
            [
                #np.random.rand(2000,8) * 0.25 + np.ones(8)*0.75,
                #np.random.rand(2000,8) * 0.1 + np.array([0,0.25,0,0,0,0,0,0]),
                #np.array([0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5])

                np.random.rand(100,2) * 0.25 + np.ones(2)*0.7,
                np.random.rand(150,2) * 0.3 + np.array([0,0.2]),
                np.random.rand(100,2) * 0.1 + np.array([0,0.2]),
                np.random.rand(40,2),
                np.array([0.5,0.5])
                ]
            )
    #testData = np.array([
    #    ### 2-D
    #    #[0.25,0.0],
    #    #[0.1,0.3],
    #    #[0.5,0.2],
    #    #[0.1,0.22],
    #    #[0.097,0.225],
    #    #[0.1,0.22],
    #    ### 3-D
    #    #[0.25,0.0,0.0],
    #    #[0.1,0.3,0.0],
    #    #[0.5,0.2,0.9],
    #    #[0.1,0.22,0.8],
    #    #[0.097,0.225,0.803],
    #    #[0.1,0.22,0.8],
    #    ### 4-D
    #    #[0.25,0.0,0.0,0.0],
    #    #[0.1,0.3,0.0,0.9],
    #    #[0.5,0.2,0.9,0.9],
    #    #[0.1,0.22,0.8,0.95],
    #    #[0.097,0.225,0.803,0.994],
    #    #[0.1,0.22,0.8,1.0],
    #    ## 8-D
    #    [0.25,0.0,0.0,0.0, 0.25,0.0,0.0,0.0],
    #    [0.1,0.3,0.0,0.9, 0.1,0.3,0.0,0.9],
    #    [0.5,0.2,0.9,0.9, 0.5,0.2,0.9,0.9],
    #    [0.1,0.22,0.8,0.95, 0.1,0.22,0.8,0.95],
    #    [0.097,0.225,0.803,0.994, 0.097,0.225,0.803,0.994],
    #    [0.1,0.22,0.8,1.0, 0.1,0.22,0.8,1.0],
    #    ])
    #testData *= (1-np.finifo( testData.dtype).resolution)
    sb = SparseBins(
            data=testData,
            minBinWidth=minBinWidth,
            maxBinWidth=maxBinWidth)
    sb.show()

    print_2d_sparseBins( sb, 'test_sparse_binning')

def print_2d_sparseBins( sb:SparseBins, outputPrefix:str):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    import os

    font={'size':12}
    mpl.rc('font',**font)
    fit, ax = plt.subplots(1,1, figsize=(5,5), tight_layout=True)
    lineEndxX = []
    lineEndxY = []
    horizontalLineYs = []
    verticalLineXs = []
    for ii in range( len( sb.binHalfWidths)):
        anchorX = sb.binCenters[ ii][ 0] - sb.binHalfWidths[ ii]
        anchorY = sb.binCenters[ ii][ 1] - sb.binHalfWidths[ ii]
        ax.add_artist( mpatches.Rectangle(
            xy=[ anchorX, anchorY],
            width = 2*sb.binHalfWidths[ ii],
            height = 2*sb.binHalfWidths[ ii],
            fill=True,
            alpha=0.2,
            ls='-',
            edgecolor='k',
            lw=2))
        #print(f"sb.data[np.array(sb.binContents[ ii]),:]: {sb.data[np.array(sb.binContents[ii]),:]}")
        points = sb.data[np.array(sb.binContents[ ii]), :]
        #print(f"points:\n{points}")
        ax.plot( points[ :, 0], points[ :, 1], lw=0, marker='.', markersize=3, color='k', alpha=0.4)
        #lineEndsXA = \
        #        [sb.binCenters[ii][0] - sb.binHalfWidths[ii],
        #         sb.binCenters[ii][0] + sb.binHalfWidths[ii]]
        #horizontalLineYA = sb.binCenters[ii][1] + sb.binHalfWidths[ii]
        #ax.hlines(
        #        y=horizontalLineYA,
        #        xmin=lineEndsXA[0],
        #        xmax=lineEndsXA[1],
        #        colors='k', linewidth=3
        #        )
        #lineEndsXB = \
        #        [sb.binCenters[ii][0] - sb.binHalfWidths[ii],
        #         sb.binCenters[ii][0] + sb.binHalfWidths[ii]]
        #horizontalLineYB = sb.binCenters[ii][1] - sb.binHalfWidths[ii]
        #ax.hlines(
        #        y=horizontalLineYB,
        #        xmin=lineEndsXB[0],
        #        xmax=lineEndsXB[1],
        #        colors='k', linewidth=3
        #        )
        ## two vertical lines per bin
        #lineEndsYC = \
        #        [sb.binCenters[ii][1] - sb.binHalfWidths[ ii],
        #         sb.binCenters[ ii][1] + sb.binHalfWidths[ ii]]
        #verticalLineXC = sb.binCenters[ii][0] + sb.binHalfWidths[ ii]
        #ax.hlines(
        #        y=verticalLineXC,
        #        xmin=lineEndsYC[0],
        #        xmax=lineEndsYC[1],
        #        colors='k', linewidth=3
        #        )
        #lineEndsYD = \
        #        [sb.binCenters[ii][1] - sb.binHalfWidths[ii],
        #         sb.binCenters[ii][1] + sb.binHalfWidths[ii]]
        #verticalLineYD = sb.binCenters[ii][0] - sb.binHalfWidths[ii]
        #ax.hlines(
        #        y=verticalLineXD,
        #        xmin=lineEndsYD[0],
        #        xmax=lineEndsYD[1],
        #        colors='k', linewidth=3
        #        )
    ax.vlines(x=0,ymin=0,ymax=1,colors='k',linewidth=1)
    ax.vlines(x=1,ymin=0,ymax=1,colors='k',linewidth=1)
    ax.hlines(y=0,xmin=0,xmax=1,colors='k',linewidth=1)
    ax.hlines(y=1,xmin=0,xmax=1,colors='k',linewidth=1)
    plt.savefig( outputPrefix + '.png')
    plt.close
    return
if __name__=="__main__":main()
