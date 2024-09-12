function myMap = occGridMapping(ranges, scanAngles, pose, param)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
% ����
%
% ÿ�׵�������
resol = param.resol;
% ��ʼ��ͼ��С��������Ϊ��λ��
myMap = zeros(param.size);
% ��ͼ��ԭ�㣨������Ϊ��λ��
origin = param.origin; 
% 
% 4. Log-odd����
lo_occ = param.lo_occ;
lo_free = param.lo_free; 
lo_max = param.lo_max;
lo_min = param.lo_min;

%-------��ʼ������--------------
lidarn = size(scanAngles,1); % ÿ��ʱ����ļ���������
N = size(ranges,2); % ʱ���������
 
for i = 1:N % ��ÿ��ʱ�������ѭ��
    theta = pose(3,i); % �����˵ķ���
    % �����˵�ʵ������
    x = pose(1,i);
    y = pose(2,i);
 
    % ʵ�������б�ռ�ݵ�ľֲ�����
    local_occs = [ranges(:,i).*cos(scanAngles+theta), -ranges(:,i).*sin(scanAngles+theta)];
 
    % ������ͼ�л����˵�����
    grid_rob = ceil(resol * [x; y]);
 
    % ���������ͼ�б�ռ�ݺͿ��е������
    for j=1:lidarn
        real_occ = local_occs(j,:) + [x, y]; % ʵ�������е�ȫ������
        grid_occ = ceil(resol * real_occ); % ������ͼ�б�ռ�ݵ������
 
        % ������ͼ�п��е�����꣨ͨ��Bresenham�㷨��
        [freex, freey] = bresenham(grid_rob(1),grid_rob(2),grid_occ(1),grid_occ(2));
        % ������ת��Ϊ�����ƫ����
        free = sub2ind(size(myMap),freey+origin(2),freex+origin(1));
        occ = sub2ind(size(myMap), grid_occ(2)+origin(2), grid_occ(1)+origin(1));
        % ���¶�����ͼ
        myMap(free) = myMap(free) - lo_free;
        myMap(occ) = myMap(occ) + lo_occ;
    end
end
% ���������Χ�����·���ֵ
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

