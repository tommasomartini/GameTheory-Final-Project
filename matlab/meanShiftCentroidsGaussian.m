function [clustCent,data2cluster,cluster2dataCell] = meanShiftCentroidsGaussian(dataPts, centroids, bandWidth, hr, hs, plotOn)
%perform MeanShift Clustering of data using a flat kernel
%
% ---INPUT---
% dataPts           - input data, (numDim x numPts) = ((RGB coords + XY coords) x numPts)
% centroids         - starting centroids
% bandWidth         - bandwidth parameter (scalar)
% hr                - bandwidth of the intensity kernel
% hs                - bandwidth of the spatial kernel
% plotOn            - plot intermediate steps
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
if nargin < 3
    error('no bandwidth specified')
end
if nargin < 6
    plotOn = 0;
end

%% Initialization
dataPts = double(dataPts);
centroids = double(centroids);

[numDim,numPts] = size(dataPts);
numClust        = 0;                        % initially no clusters
bandSq          = bandWidth^2;              % square of the bandwidth
initPtInds      = 1 : numPts;               % indexes of the initial points (as many initial points as the input data points)
numInitPts      = numPts;                   % number of points to possibly use as initilization points
% maxPos          = max(dataPts,[],2);        % biggest size in each dimension (largest value for each row)
% minPos          = min(dataPts,[],2);        % smallest size in each dimension (smallest value for each row)
% boundBox        = maxPos-minPos;            % bounding box size
% sizeSpace       = norm(boundBox);           % indicator of size of data space
stopThresh      = 1e-3*bandWidth;           % when mean has converged
clustCent       = [];                       % centers of the clusters
beenVisitedFlag = zeros(1,numPts,'uint8');  % tracks if a point has been already seen
clusterVotes    = zeros(1,numPts,'uint16'); % used to resolve conflicts on cluster membership

num_centroids   = size(centroids, 2);       % number of centroids provided

pause on;
%% Main cycle

weight_map = exp(-(0 : 255^2) / hr^2);  % map of the color weights

num_centroids_left = num_centroids; % how many centroids I don't have yet used
centroids_left = centroids;
centroid_counter = 0;
while num_centroids_left    % while there are still some centroids

    % rand is a double in the open interval (0, 1). 
    % The vector initPointInds contains the indeces of the data points not
    % yet visited. A point has been visited if it has given its
    % contribution to evaluating a mean
%     tempInd         = ceil((num_centroids_left - 1e-6) * rand); % pick a random index between 1 and numInitPoints
%     stInd           = centroids_indeces(tempInd);              % pick the corresponding index
%     myMean          = centroids_left(:, stInd);                % intialize mean to this point's location
    centroid_counter = centroid_counter + 1;
    myMean          = centroids(:, centroid_counter);
    myMembers       = [];                               % points that will get added to this cluster                          
    thisClusterVotes    = zeros(1,numPts,'uint16');     % used to resolve conflicts on cluster membership

    while 1     % loop untill convergence
        
        sqColorDistToAll = sum((repmat(myMean(1 : 3, :), 1, numPts) - dataPts(1 : 3, :)).^2);
        sqSpaceDistToAll = sum((repmat(myMean(2 : 2, :), 1, numPts) - dataPts(2 : 4, :)).^2);
        
        sqDistToAll = sum((repmat(myMean,1,numPts) - dataPts).^2);    % squared distance from mean to all points still active
%         inInds      = find(sqColorDistToAll < bandSq); 
        inInds      = find(sqDistToAll < bandSq);               % indeces of points whose distance is under the square of the bandwidth 
        thisClusterVotes(inInds) = thisClusterVotes(inInds)+1;  % add a vote for all the in points belonging to this cluster
        
%         kerColorDistToAll = exp(-sqColorDistToAll ./ hr^2);
%         kerSpaceDistToAll = exp(-sqSpaceDistToAll ./ hs^2);
%         totalKernel = kerColorDistToAll .* kerSpaceDistToAll;
        
        myOldMean   = myMean;                       % save the old mean
        tmpWeightedPts = 
        myMean      = mean(dataPts(:,inInds), 2);   % compute the new mean: mean of the points "close enough"
        
        myMembers   = [myMembers inInds];           % add all the points within bandWidth to the cluster
        beenVisitedFlag(myMembers) = 1;             % mark that these points have been visited
        
        if plotOn
            figure(111),clf,hold on
            if numDim == 2
                plot(dataPts(1,:),dataPts(2,:),'.')
                plot(dataPts(1,myMembers),dataPts(2,myMembers),'ys')
                plot(myMean(1),myMean(2),'go')
                plot(myOldMean(1),myOldMean(2),'rd')
                pause(0.1);
            elseif numDim == 3
                scatter3(dataPts(1,:),dataPts(2,:),dataPts(3,:),'.')
                scatter3(dataPts(1,myMembers),dataPts(2,myMembers),dataPts(3,myMembers),'ys')
                scatter3(myMean(1),myMean(2),myMean(3),'go')
                scatter3(myOldMean(1),myOldMean(2),myOldMean(3),'rd')
                view(40,35)
                pause(0.1);
            end
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
    
    centroids_left = centroids_left(:, 2 : end);
    num_centroids_left = num_centroids_left - 1;
    
    initPtInds      = find(beenVisitedFlag == 0);   % we can initialize with any of the points not yet visited
    numInitPts      = length(initPtInds);           % number of active points in set
end







bandWidth = bandWidth * 3;
bandSq = bandWidth^2;  

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
        sqColorDistToAll = sum((repmat(myMean(1 : 3, :), 1, numPts) - dataPts(1 : 3, :)).^2);
        sqSpaceDistToAll = sum((repmat(myMean(2 : 2, :), 1, numPts) - dataPts(2 : 4, :)).^2);
        kerColorDistToAll = exp(-sqColorDistToAll ./ hr^2);
        kerSpaceDistToAll = exp(-sqSpaceDistToAll ./ hs^2);
        totalKernel = kerColorDistToAll .* kerSpaceDistToAll;
         
        sqDistToAll = sum((repmat(myMean,1,numPts) - dataPts).^2);    % squared distance from mean to all points still active
        % inInds      = find(totalKernel < bandSq);
        inInds      = find(sqDistToAll < bandSq);               % indeces of points whose distance is under the square of the bandwidth 
        thisClusterVotes(inInds) = thisClusterVotes(inInds)+1;  % add a vote for all the in points belonging to this cluster
        
        myOldMean   = myMean;                       % save the old mean
        myMean      = mean(dataPts(:,inInds), 2);   % compute the new mean: mean of the points "close enough"
        myMembers   = [myMembers inInds];           % add all the points within bandWidth to the cluster
        beenVisitedFlag(myMembers) = 1;             % mark that these points have been visited
        
        if plotOn
            figure(112),clf,hold on
            if numDim == 2
                plot(dataPts(1,:),dataPts(2,:),'.')
                plot(dataPts(1,myMembers),dataPts(2,myMembers),'ys')
                plot(myMean(1),myMean(2),'go')
                plot(myOldMean(1),myOldMean(2),'rd')
                pause(0.1);
            elseif numDim == 3
                scatter3(dataPts(1,:),dataPts(2,:),dataPts(3,:),'.')
                scatter3(dataPts(1,myMembers),dataPts(2,myMembers),dataPts(3,myMembers),'ys')
                scatter3(myMean(1),myMean(2),myMean(3),'go')
                scatter3(myOldMean(1),myOldMean(2),myOldMean(3),'rd')
                view(40,35)
                pause(0.1);
            end
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
% [~,data2cluster] = max(clusterVotes,[],1);    % a point belongs to the cluster with the most votes
[~,data2cluster] = max(clusterVotes,[],1);

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



