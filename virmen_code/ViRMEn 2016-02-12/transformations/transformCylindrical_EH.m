function coords2D = transformCylindrical_EH(coords3D)


% viewAngle = 112.5*pi/180; % how far beyond 90 degrees to the left and right the screen is behind the mouse
viewAngle = 90*pi/180; % 
backwardAngle = viewAngle - pi/2;

% scr = getScreenSize();%%get(0,'screensize');
scr_width = 3840; %total screen width in pix
% scr_height = 1080; 
scr_height = 1080; 
scr = [1 1 scr_width scr_height];

aspectRatio = (scr(3)/scr(4));
coords2D = zeros(size(coords3D));

% calculate horizontal angle theta for each point. theta begins at -0 and goes clockwise to +2viewAngle
theta = atan2(coords3D(1,:),coords3D(2,:))+viewAngle;

% calculate screen x for each point
coords2D(1,:) = aspectRatio./(viewAngle).*theta - aspectRatio;

% calculate the distance in the xy plane to each point
r=(coords3D(1,:).^2 + coords3D(2,:).^2).^.5;

% calculate the y screen coordinate for each point
coords2D(2,:) = coords3D(3,:)./r;




% set all points to true
coords2D(3,:) = true;


% clip stuff left and right of the screen
f = find(abs(coords2D(1,:))>aspectRatio);
coords2D(1,f) = sign(coords2D(1,f))*aspectRatio;
coords2D(3,f) = false;

% clip stuff above and below the screen
f = find(abs(coords2D(2,:))>1);
coords2D(2,f) = sign(coords2D(2,f));
coords2D(3,f) = false;
