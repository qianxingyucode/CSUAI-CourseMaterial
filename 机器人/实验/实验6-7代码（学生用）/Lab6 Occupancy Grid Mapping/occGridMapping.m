function myMap = occGridMapping(ranges, scanAngles, pose, param)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
% 参数
%
% 每米的网格数
resol = param.resol;
% 初始地图大小（以像素为单位）
myMap = zeros(param.size);
% 地图的原点（以像素为单位）
origin = param.origin; 
% 
% 4. Log-odd参数
lo_occ = param.lo_occ;
lo_free = param.lo_free; 
lo_max = param.lo_max;
lo_min = param.lo_min;

%-------初始化代码--------------
lidarn = size(scanAngles,1); % 每个时间戳的激光束数量
N = size(ranges,2); % 时间戳的数量
 
for i = 1:N % 对每个时间戳进行循环
    theta = pose(3,i); % 机器人的方向
    % 机器人的实际坐标
    x = pose(1,i);
    y = pose(2,i);
 
    % 实际世界中被占据点的局部坐标
    local_occs = [ranges(:,i).*cos(scanAngles+theta), -ranges(:,i).*sin(scanAngles+theta)];
 
    % 度量地图中机器人的坐标
    grid_rob = ceil(resol * [x; y]);
 
    % 计算度量地图中被占据和空闲点的坐标
    for j=1:lidarn
        real_occ = local_occs(j,:) + [x, y]; % 实际世界中的全局坐标
        grid_occ = ceil(resol * real_occ); % 度量地图中被占据点的坐标
 
        % 度量地图中空闲点的坐标（通过Bresenham算法）
        [freex, freey] = bresenham(grid_rob(1),grid_rob(2),grid_occ(1),grid_occ(2));
        % 将坐标转换为数组的偏移量
        free = sub2ind(size(myMap),freey+origin(2),freex+origin(1));
        occ = sub2ind(size(myMap), grid_occ(2)+origin(2), grid_occ(1)+origin(1));
        % 更新度量地图
        myMap(free) = myMap(free) - lo_free;
        myMap(occ) = myMap(occ) + lo_occ;
    end
end
% 如果超出范围则重新分配值
myMap(myMap < lo_min) = lo_min;
myMap(myMap > lo_max) = lo_max;

%     % Find occupied-measurement cells and free-measurement cells
%       distances/ranges
% 
%     % Update the log-odds
%      -calculate
% 
%     % Saturate the log-odd values
%     
% 
%     % Visualize the map as needed

end

