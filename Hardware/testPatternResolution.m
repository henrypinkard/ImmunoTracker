function [] = testPatternResolution()

NUMROWS = 8;
NUMCOLS = 8;

[rowIndices, colIndices] = meshgrid(0:NUMROWS-1,0:NUMCOLS-1);

sampleVals = @(nRows,nCols,rIndices,cIndices) sqrt((rIndices./(nRows-1)).^2 + (cIndices./(nCols-1)).^2);
values = sampleVals(NUMROWS,NUMCOLS,rowIndices,colIndices);


[queryPointsR, queryPointsC] = meshgrid(linspace(0,NUMROWS-1,200),linspace(0,NUMCOLS-1,200));

qpLowerRow = floor(queryPointsR);
qpLowerRowWeight = 1 - (queryPointsR - qpLowerRow);
qpLowerCol = floor(queryPointsC);
qpLowerColWeight = 1- ( queryPointsC - qpLowerCol);

surf(rowIndices,colIndices,values);
colormap viridis

hold on
interpValsLCLR = sampleVals(NUMROWS,NUMCOLS,qpLowerRow,qpLowerCol);
interpValsUCLR = sampleVals(NUMROWS,NUMCOLS,qpLowerRow,qpLowerCol+1);
interpValsLCUR = sampleVals(NUMROWS,NUMCOLS,qpLowerRow+1,qpLowerCol);
interpValsUCUR = sampleVals(NUMROWS,NUMCOLS,qpLowerRow+1,qpLowerCol+1);
weightedInterpLR = interpValsLCLR.*qpLowerColWeight + interpValsUCLR.*(1-qpLowerColWeight);
weightedInterpUR = interpValsLCUR.*qpLowerColWeight + interpValsUCUR.*(1-qpLowerColWeight);
weightedInterp = weightedInterpLR.*qpLowerRowWeight + weightedInterpUR.*(1-qpLowerRowWeight);

surf(queryPointsR,queryPointsC,weightedInterp);
hold off
alpha(0.5)

% [ lowerBoundIndex, weights ] = calcLUT( NUMCOLS );

end

function [ lowerBoundIndex, weights ] = calcLUT( NUM_COLS )
PHASE_SHIFT_US = 2.4;
TIME_PER_ROW_US = 126.2;
LUT_SIZE = 127;
normalizedColSize = 1.0 /  NUM_COLS;
 lowerBoundIndex = [];
 weights = [];
  for  time_us = 1:LUT_SIZE 
    spatialPos = min((1 + cos((time_us + PHASE_SHIFT_US) / TIME_PER_ROW_US * 2 * 3.141592))/2.0,0.9999999);
    lowerBoundIndex(time_us) = (spatialPos * (NUM_COLS - 1));
    weights(time_us) = 1.0 -  mod(spatialPos,normalizedColSize) / normalizedColSize;
  end
end