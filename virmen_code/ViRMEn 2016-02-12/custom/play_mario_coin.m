function play_mario_coin()
    [y,Fs] = audioread('Mario-coin-sound.mp3');
    sound(y.*0.1,Fs);
end