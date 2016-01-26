function joints = annolist2matrix(annolist)

joints = nan(2,14,length(annolist));
nJointsAnnolist = 16;

for imgidx = 1:length(annolist)
    points = annolist(imgidx).annorect.annopoints.point;
    pointsAll = nan(nJointsAnnolist,2);
    for kidx = 0:nJointsAnnolist-1
        p = util_get_annopoint_by_id(points,kidx);
        if (~isempty(p))
            pointsAll(kidx+1,:) = [p.x p.y];
        end
    end
    joints(:,1:6,imgidx) = pointsAll(1:6,:)';
    joints(:,7:12,imgidx) = pointsAll(11:16,:)';
    joints(:,13:14,imgidx) = pointsAll(9:10,:)';
end

end