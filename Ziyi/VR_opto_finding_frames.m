opto_duration = 2;
frame_rate = 31.25;
opto_frames = opto_duration *frame_rate;
optoEventsIdx = VR.optoEventIdx;
optoEvents = zeros(size(VR.imageSync));
for i = 1:length(optoEventsIdx)
    idx = optoEventsIdx(i);
    optoEvents(idx:idx + opto_frames - 1) = 1;
end