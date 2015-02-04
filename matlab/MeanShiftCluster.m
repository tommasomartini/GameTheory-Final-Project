function [clustCent,data2cluster,cluster2dataCell] = MeanShiftCluster(dataPts,bandWidth)
%perform MeanShift Clustering of data using a flat kernel
%
% ---INPUT---
% dataPts           - input data, (numDim x numPts)
% bandWidth         - is bandwidth parameter (scalar)
% ---OUTPUT---
% clustCent         - locations of cluster centers (numDim x numClust)
% data2cluster      - for every data point which cluster it belongs to (numPts)
% cluster2dataCell  - for every cluster which points are in it (numClust)
% 
% Bryan Feldman 02/24/06
% MeanShift first appears in
% K. Funkunaga and L.D. Hosteler, "The Estimation of the Gradient of a
% Density Function, with Applications in Pattern Recognition"

%% Check input
if nargin < 2
    error('no bandwidth specified')
end

%% Initialization
[numDim,numPts] = size(dataPts);
numClust        = 0;                        % initially no clusters
bandSq          = bandWidth^2;              % square of the bandwidth
initPtInds      = 1 : numPts;               % indexes of the initial points (as many initial points as the input data points)
numInitPts      = numPts;                   % number of points to possibly use as initilization points
maxPos          = max(dataPts,[],2);        % biggest size in each dimension (largest value for each row)
minPos          = min(dataPts,[],2);        % smallest size in each dimension (smallest value for each row)
boundBox        = maxPos-minPos;            % bounding box size
sizeSpace       = norm(boundBox);           % indicator of size of data space
stopThresh      = 1e-3*bandWidth;           % when mean has converged
clustCent       = [];                       % centers of the clusters
beenVisitedFlag = zeros(1,numPts,'uint8');  % tracks if a point has been already seen
clusterVotes    = zeros(1,numPts,'uint16'); % used to resolve conflicts on cluster membership

pause on;
%% Main cycle

while numInitPts    % while there are still some initPts

    % rand is a double in the open interval (0, 1). 
    % The vector initPointInds contains the indeces of the data points not
    % yet visited. A point has been visited if it has given its
    % contribution to evaluating a mean
    tempInd         = ceil((numInitPts - 1e-6) * rand); % pick a random index between 1 and numInitPoints
    stInd           = initPtInds(tempInd);              % pick the corresponding index
    myMean          = dataPts(:, stInd);                % intialize mean to this point's location
    myMembers       = [];                               % points that will get added to this cluster                          
    thisClusterVotes    = zeros(1,numPts,'uint16');     % used to resolve conflicts on cluster membership

    while 1     % loop untill convergence
        
        sqDistToAll = sum((repmat(myMean,1,numPts) - dataPts).^2);    % squared distance from mean to all points still active
        inInds      = find(sqDistToAll < bandSq);               % indeces of points whose distance is under the square of the bandwidth 
        thisClusterVotes(inInds) = thisClusterVotes(inInds)+1;  % add a vote for all the in points belonging to this cluster
        
        myOldMean   = myMean;                       % save the old mean
        myMean      = mean(dataPts(:,inInds), 2);   % compute the new mean: mean of the points "close enough"
        myMembers   = [myMembers inInds];           % add all the points within bandWidth to the cluster
        beenVisitedFlag(myMembers) = 1;             % mark that these points have been visited
        
        figure(112),clf,hold on
        if numDim == 2
            plot(dataPts(1,:),dataPts(2,:),'.')
            plot(dataPts(1,myMembers),dataPts(2,myMembers),'ys')
            plot(myMean(1),myMean(2),'go')
            plot(myOldMean(1),myOldMean(2),'rd')
            pause(0.1);
        end

        if norm(myMean - myOldMean) < stopThresh    % mean does not move much
            
            %check for merge possibilities
            mergeWith = 0;  % index of the cluster to merge with
            for cN = 1 : numClust   % for each found cluster
                distToOther = norm(myMean - clustCent(:,cN));   % distance between the current mean and the center of the cluster
                if distToOther < bandWidth/2                    % if its within bandwidth/2 merge new and old
                    mergeWith = cN;
                    % if I find a cluster I can merge with, I keep that one
                    % and will not look for any other. The first one I find
                    % is OK!
                    break;
                end
            end
            
            
            if mergeWith > 0    % something to merge
                clustCent(:,mergeWith)       = 0.5*(myMean+clustCent(:,mergeWith));             % the new center of the cluster I merge with is the mean between its center and the current mean
%                 clustMembsCell{mergeWith}    = unique([clustMembsCell{mergeWith} myMembers]);   % record which points inside 
                clusterVotes(mergeWith, :)   = clusterVotes(mergeWith,:) + thisClusterVotes;    % add these votes to the merged cluster
            else    % it is a new cluster
                numClust                    = numClust + 1;         % increment the number of clusters
                clustCent(:, numClust)      = myMean;              % record the mean as the center of this new cluster
                %clustMembsCell{numClust}    = myMembers;                    %store my members
                clusterVotes(numClust, :)   = thisClusterVotes;
            end

            break;
        end

    end
    
    initPtInds      = find(beenVisitedFlag == 0);   % we can initialize with any of the points not yet visited
    numInitPts      = length(initPtInds);           % number of active points in set

end

% clusterVotes is a matrix with n_rows = n_data_points and n_cols =
% n_clusters. Every data point votes for one or more cluster
[val,data2cluster] = max(clusterVotes,[],1);    % a point belongs to the cluster with the most votes

%% Output
% If they want the cluster2data cell find it for them
if nargout > 2
    cluster2dataCell = cell(numClust,1);
    for cN = 1:numClust
        myMembers = find(data2cluster == cN);
        cluster2dataCell{cN} = myMembers;
    end
end

pause off;


