function RV = GenerateRVs_both(Q,T)

f_rho = zeros(size(T,1) - size(Q,1) +1,size(T,2) - size(Q,2) +1,size(T,3) - size(Q,3) +1);
f_rho1 = zeros(size(T,1) - size(Q,1) +1,size(T,2) - size(Q,2) +1,size(T,3) - size(Q,3) +1);

space = 2;
space_t = 1;

temp = Q(1:space:end,1:space:end,1:space_t:end,:);
temp1 = Q(1:space:end,end:-space:1,1:space_t:end,:);

F_Q = temp(:);
F_Q1 = temp1(:);

norm_FQ = norm(F_Q(:),'fro');

for i = 1:size(T,1) - size(Q,1) +1
    for j = 1:size(T,2) - size(Q,2) +1
        for k = 1:size(T,3) - size(Q,3) + 1
            F_T = T(i:space:i+size(Q,1)-1, j:space:j+size(Q,2)-1,k:space_t:k+size(Q,3)-1,:);
            F_T_i = F_T(:);
            den = (norm_FQ*norm(F_T_i(:),'fro'));
            rho = F_Q(:)'*F_T_i(:)/den;
            rho1 = F_Q1(:)'*F_T_i(:)/den;
            f_rho(i,j,k) = (rho^2)/(1-rho^2);
            f_rho1(i,j,k) = (rho1^2)/(1-rho1^2);
        end
    end
    i
end

f_rho = max(cat(4,f_rho,f_rho1),[],4);    

RV = zeros(size(T,1),size(T,2),size(T,3)); % Initialize RV
RV(floor(size(Q,1)/2):floor(size(Q,1)/2)+size(f_rho,1)-1,floor(size(Q,2)/2):floor(size(Q,2)/2)+size(f_rho,2)-1,...
    floor(size(Q,3)/2):floor(size(Q,3)/2)+size(f_rho,3)-1) = f_rho;
