function [ybinned_circ] = convert_ypos_to_circular_coords_per_ep(ybinned, rewlocs, eps, track_length)

ypos_circ = zeros(size(ybinned));
for ep=1:length(rewlocs)-1
    ypos = ybinned(eps(ep):eps(ep+1));
    rewloc = rewlocs(ep);
    ypos_circ(eps(ep):eps(ep+1)) = convert_to_circular_coordinates(ypos, rewloc, track_length);
end
end