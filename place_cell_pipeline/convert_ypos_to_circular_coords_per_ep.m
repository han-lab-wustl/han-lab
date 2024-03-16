function [ypos_circ] = convert_ypos_to_circular_coords_per_ep(ybinned, rewlocs, eps, track_length)

ypos_circ = zeros(size(ybinned));
if length(rewlocs) > 1
    for ep=1:length(eps)-1
        ypos = ybinned(eps(ep):eps(ep+1));
        rewloc = rewlocs(ep);
        ypos_circ(eps(ep):eps(ep+1)) = convert_to_circular_coordinates(ypos, rewloc, track_length);
    end
else
    ypos = ybinned;
    rewloc = rewlocs;
    ypos_circ = convert_to_circular_coordinates(ypos, rewloc, track_length);
end

end