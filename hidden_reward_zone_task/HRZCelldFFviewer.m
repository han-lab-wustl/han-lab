function varargout = HRZCelldFFviewer(varargin)
% HRZCELLDFFVIEWER MATLAB code for HRZCelldFFviewer.fig
%      HRZCELLDFFVIEWER, by itself, creates a new HRZCELLDFFVIEWER or raises the existing
%      singleton*.
%
%      H = HRZCELLDFFVIEWER returns the handle to a new HRZCELLDFFVIEWER or the handle to
%      the existing singleton*.
%
%      HRZCELLDFFVIEWER('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in HRZCELLDFFVIEWER.M with the given input arguments.
%
%      HRZCELLDFFVIEWER('Property','Value',...) creates a new HRZCELLDFFVIEWER or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before HRZCelldFFviewer_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to HRZCelldFFviewer_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help HRZCelldFFviewer

% Last Modified by GUIDE v2.5 27-Jun-2023 15:19:56

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @HRZCelldFFviewer_OpeningFcn, ...
                   'gui_OutputFcn',  @HRZCelldFFviewer_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before HRZCelldFFviewer is made visible.
function HRZCelldFFviewer_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to HRZCelldFFviewer (see VARARGIN)

% Choose default command line output for HRZCelldFFviewer
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes HRZCelldFFviewer wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = HRZCelldFFviewer_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on selection change in popupmenu1.
function popupmenu1_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu1
W = evalin('base','whos');
doesAexist = ismember('alldata',[W(:).name]);
if doesAexist
    alldata = evalin('base','alldata');
    changerews = find(alldata.changeRewLoc);
    if handles.popupmenu1.Value == 1
        set(handles.axes1,'Xlim',[0 alldata.timedFF(end)])
    elseif handles.popupmenu1.Value == length(handles.popupmenu1.String)
        set(handles.axes1,'Xlim',[alldata.timedFF(changerews(end)) alldata.timedFF(end)])
    else
        set(handles.axes1,'Xlim',[alldata.timedFF(changerews(handles.popupmenu1.Value-1)) alldata.timedFF(changerews(handles.popupmenu1.Value))])
    end
end


% --- Executes during object creation, after setting all properties.
function popupmenu1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in popupmenu2.
function popupmenu2_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu2 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu2
W = evalin('base','whos');
doesAexist = ismember('alldata',[W(:).name]);
if doesAexist
    alldata = evalin('base','alldata');
    handles.popupmenu2.String = string(1:size(alldata.all.dff,1));
    
    handles.slider1.Min = 1;
    handles.slider1.Max = size(alldata.all.dff,1);
    if handles.slider1.Value ~= handles.popupmenu2.Value
    handles.slider1.Value = handles.popupmenu2.Value;
    end
    
    hold(handles.axes2,'on')
    scatter(alldata.timedFF,alldata.ybinned,2,[0.5 0.5 0.5],'filled','Parent',handles.axes2)
    scatter(alldata.timedFF,alldata.ybinned,10,(alldata.all.dff(handles.popupmenu2.Value,:)),'filled','MarkerFaceAlpha',0.7,'Parent',handles.axes2)
    colormap(handles.axes2,flipud(bone))
    hold(handles.axes2,'off')
    
    plot(alldata.timedFF,alldata.all.dff(handles.popupmenu2.Value,:),'Color',[0 0 0.5],'Parent',handles.axes3)
    
    hold(handles.axes4,'off')
    truecells = find(alldata.iscell(:,1)==1);
    truecells(logical(alldata.remove_iscell)) = [];
    imagesc(alldata.ops.meanImg,'Parent',handles.axes4)
    hold(handles.axes4,'on')
    colormap(handles.axes4,bone)
    scatter(alldata.stat{truecells(handles.popupmenu2.Value)}.xpix,alldata.stat{truecells(handles.popupmenu2.Value)}.ypix,1,'filled','y','Parent',handles.axes4)
    
    
    hold(handles.axes5,'off')
    truecells = find(alldata.iscell(:,1)==1);
    truecells(logical(alldata.remove_iscell)) = [];
    imagesc(alldata.ops.meanImg,'Parent',handles.axes5)
    hold(handles.axes5,'on')
    colormap(handles.axes5,bone)
    scatter(alldata.stat{truecells(handles.popupmenu2.Value)}.xpix,alldata.stat{truecells(handles.popupmenu2.Value)}.ypix,1,'filled','y','Parent',handles.axes5)
    xlim([min(alldata.stat{truecells(handles.popupmenu2.Value)}.xpix)-10 max(alldata.stat{truecells(handles.popupmenu2.Value)}.xpix)+10])
    ylim([min(alldata.stat{truecells(handles.popupmenu2.Value)}.ypix)-10 max(alldata.stat{truecells(handles.popupmenu2.Value)}.ypix)+10])
    
    
    handles.slider2.Max = max(alldata.all.dff(handles.popupmenu2.Value,:));
    handles.slider2.Min = min(alldata.all.dff(handles.popupmenu2.Value,:));
    handles.slider3.Max = max(alldata.all.dff(handles.popupmenu2.Value,:));
    handles.slider3.Min = min(alldata.all.dff(handles.popupmenu2.Value,:));
    
    handles.slider2.Value = max(alldata.all.dff(handles.popupmenu2.Value,:));
    handles.slider3.Value = min(alldata.all.dff(handles.popupmenu2.Value,:));
    
    handles.slider2.SliderStep = [ 1/100 1/100]*range(alldata.all.dff(handles.popupmenu2.Value,:));
    handles.slider3.SliderStep = [ 1/100 1/100]*range(alldata.all.dff(handles.popupmenu2.Value,:));
    
    
    
    guidata(hObject,handles)
    slider1_Callback(hObject, eventdata, handles)
    popupmenu1_Callback(hObject, eventdata, handles)
    checkbox1_Callback(hObject, eventdata, handles)
    
end


% --- Executes during object creation, after setting all properties.
function popupmenu2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on slider movement.
function slider1_Callback(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
handles.slider1.Value = round(handles.slider1.Value);
W = evalin('base','whos');
doesAexist = ismember('alldata',[W(:).name]);
if doesAexist
    if handles.popupmenu2.Value ~= handles.slider1.Value
    handles.popupmenu2.Value = handles.slider1.Value;
    guidata(hObject,handles)
    popupmenu2_Callback(hObject, eventdata, handles)
    end
end


% --- Executes during object creation, after setting all properties.
function slider1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[filename,pathname] = uigetfile('*.mat','Select the analyzed Fall file');
alldata = load([pathname filename]);
assignin('base','alldata',alldata);
handles.popupmenu1.Value = 1;
handles.popupmenu2.Value = 1;
handles.popupmenu1.String = ["All" string(1:length(find(alldata.changeRewLoc)))];
plot(alldata.timedFF,alldata.ybinned/alldata.VR.scalingFACTOR,'Color',[0.5 0.5 0.5],'Parent',handles.axes1)
hold(handles.axes1,'on')
plot(alldata.timedFF(find(alldata.licks)),alldata.ybinned(find(alldata.licks))/alldata.VR.scalingFACTOR,'.','Color',[0 0 0.3],'Parent',handles.axes1)
plot(alldata.timedFF(find(alldata.rewards)),alldata.ybinned(find(alldata.rewards))/alldata.VR.scalingFACTOR,'.','Color',[0.8 0 0],'Parent',handles.axes1)
linkaxes([handles.axes1,handles.axes2,handles.axes3],'x')
set(handles.slider1,'Min',1)
set(handles.slider1,'Max',size(alldata.all.dff,1))
 set(handles.slider1, 'SliderStep', [1/size(alldata.all.dff,1), 1/size(alldata.all.dff,1) ]);
guidata(hObject,handles)
popupmenu1_Callback(hObject, eventdata, handles)
popupmenu2_Callback(hObject, eventdata, handles)

function UIFigureKeyPress(hObject, eventdata, handles)
	value = handles.slider1.Value; % get the slider value
	key = event.Key; % get the pressed key value
        if strcmp(key,'leftarrow')
        	value = value-1; % left value
        elseif strcmp(key,'rightarrow')
                value = value+1; % right value
        end
        handles.slider1.Value = value; % set the slider value
	guidata(hObject,handles) % execute the slider callback


% --- Executes on slider movement.
function slider2_Callback(hObject, eventdata, handles)
% hObject    handle to slider2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
W = evalin('base','whos');
doesAexist = ismember('alldata',[W(:).name]);
if doesAexist
    if handles.slider2.Value<= handles.slider3.Value
        handles.slider2.Value = handles.slider2.Value+2*handles.slider2.SliderStep(1);
    end
    if handles.slider2.Value > handles.slider3.Value
     caxis(handles.axes2,[handles.slider3.Value handles.slider2.Value])
    else
        handles.slider2.Value = handles.slider3.Value+2*handles.slider2.SliderStep(1);
    end
    if handles.checkbox1.Value
     checkbox1_Callback(hObject, eventdata, handles)
    end
end


% --- Executes during object creation, after setting all properties.
function slider2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on slider movement.
function slider3_Callback(hObject, eventdata, handles)
% hObject    handle to slider3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
W = evalin('base','whos');
doesAexist = ismember('alldata',[W(:).name]);
if doesAexist
    if handles.slider3.Value>= handles.slider2.Value
        handles.slider3.Value = handles.slider3.Value-2*handles.slider3.SliderStep(1);
    end
    if handles.slider3.Value < handles.slider2.Value
    caxis(handles.axes2,[handles.slider3.Value handles.slider2.Value])
    else
        handles.slider3.Value = handles.slider3.Value-2*handles.slider2.SliderStep(1);
    end
    if handles.checkbox1.Value
    checkbox1_Callback(hObject, eventdata, handles)
    end
end


% --- Executes during object creation, after setting all properties.
function slider3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on button press in checkbox1.
function checkbox1_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox1
deleteindx = [];
for cs = 1:length(handles.axes3.Children)
    if handles.axes3.Children(cs).Color(1) == 1
        deleteindx = [deleteindx cs];
    end
end
delete(handles.axes3.Children(deleteindx))
if handles.checkbox1.Value
    hold(handles.axes3,'on')
    xlims = xlim(handles.axes3);
    plot(xlims,[handles.slider2.Value handles.slider2.Value],'r-','Parent',handles.axes3)
    plot(xlims, [handles.slider3.Value handles.slider3.Value],'r-','Parent',handles.axes3)
    hold(handles.axes3,'off')
    
end
