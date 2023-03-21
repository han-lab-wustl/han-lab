function format_conversion_Suite2p_CNMF_e(file_path)
%% Converting the outputs from CNMF-e or Suite2p to the required input format for CellReg

input_format = 'Suite2p'; % can also be CNMF
%runs for a single session...
disp(file_path);
this_session_data=load(file_path);

if strcmp(input_format,'Suite2p')
    this_session_x_size=this_session_data.ops.Lx+1;
    this_session_y_size=this_session_data.ops.Ly+1;
    this_session_num_cells=size(this_session_data.stat,2);
    this_session_converted_footprints=zeros(this_session_num_cells,this_session_y_size,this_session_x_size);
    for k=1:this_session_num_cells
        for l=1:length(this_session_data.stat{k}.ypix)
            this_session_converted_footprints(k,this_session_data.stat{k}.ypix(l)+1,this_session_data.stat{k}.xpix(l)+1)=this_session_data.stat{k}.lam(l);
            
        end
    end
    
elseif strcmp(input_format','CNMF-e')
    this_session_y_size=this_session_data.neuron.options.d1;
    this_session_x_size=this_session_data.neuron.options.d2;
    this_session_num_cells=size(this_session_data.neuron.A,2);
    this_session_converted_footprints=zeros(this_session_num_cells,this_session_y_size,this_session_x_size);
    for k=1:this_session_num_cells
        this_session_converted_footprints(k,:,:)=reshape(this_session_data.neuron.A(:,k),this_session_y_size,this_session_x_size);
    end
end

%separate into file parts to save
[f,n,e]=fileparts(file_path);
save(sprintf(['%s\\', 'converted_%s%s'], f,n,e),'this_session_converted_footprints','-v7.3')


end