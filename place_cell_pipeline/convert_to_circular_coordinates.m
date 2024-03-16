function centered_coordinates = convert_to_circular_coordinates(coordinates, center_location, track_length)
% Convert track coordinates from 0 to track_length (default: 270 cm) to -pi to pi radians,
% centered at a specified location.
%
% Args:
%   coordinates (double array): 1D array of track coordinates in cm.
%   center_location (double): Location to center the coordinates at, in cm.
%   track_length (double, optional): Length of the track in cm (default: 270).
%
% Returns:
%   centered_coordinates (double array): Converted coordinates in radians, centered at the specified location.

if nargin < 3
    track_length = 270; % Default track length is 270 cm
end

% Convert coordinates and center_location to radians
coordinates_radians = coordinates * (2 * pi / track_length);
center_radians = center_location * (2 * pi / track_length);

% Center coordinates_radians around center_radians
centered_coordinates_radians = coordinates_radians - center_radians;

% Wrap the centered_coordinates_radians to -pi to pi range
centered_coordinates_radians = mod(centered_coordinates_radians + pi, 2 * pi) - pi;

centered_coordinates = centered_coordinates_radians;
end