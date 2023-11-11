function h=find_figure(figname)

h = findobj('Tag', figname); % check for pre-existing window
if(isempty(h)) % if none, make one
	h = figure('Tag', figname, 'Name', figname, 'NumberTitle', 'off');
end
figure(h);
