clear
fid1 = fopen('log_test.dat');
S1 = textscan(fid1,'%n%s%n%n%n');
fclose(fid1);
M = S1{5};    % MASK 
fid2 = fopen('log_metrics.dat');
S2 = textscan(fid2,'%s%n%n%n');
fclose(fid2);
C = S2{2};    % PSNR_metrics
D = S2{3};    % SSIM
E = S2{4};    % MAE_metrics
n = length(M);
N1 = []; N2 = []; N3 = []; N4 = []; N5 = []; N6 = []; N7 = [];
for i = 1 : n
    if M(i) < 0.1
        N1 = [N1,i];
    else if M(i) >= 0.1 && M(i) < 0.2
            N2 = [N2,i];
        else if M(i) >= 0.2 && M(i) < 0.3
                N3 = [N3,i];
            else if M(i) >= 0.3 && M(i) < 0.4
                    N4 = [N4,i];
                else if M(i) >= 0.4 && M(i) < 0.5
                        N5 = [N5,i];
                    else if M(i) >= 0.5 && M(i) < 0.6
                            N6 = [N6,i];
                        else if M(i) >= 0.6 && M(i) < 0.7
                                N7 = [N7,i];
                            end
                        end
                    end
                end
            end
        end
    end
end
C1 = mean(C(N1));
C2 = mean(C(N2));
C3 = mean(C(N3));
C4 = mean(C(N4));
C5 = mean(C(N5));
C6 = mean(C(N6));
C7 = mean(C(N7));
D1 = mean(D(N1));
D2 = mean(D(N2));
D3 = mean(D(N3));
D4 = mean(D(N4));
D5 = mean(D(N5));
D6 = mean(D(N6));
D7 = mean(D(N7));
E1 = mean(E(N1));
E2 = mean(E(N2));
E3 = mean(E(N3));
E4 = mean(E(N4));
E5 = mean(E(N5));
E6 = mean(E(N6));
E7 = mean(E(N7));
%disp('Average(mask 0-10%): PSNR_test; MAE_test; PSNR_metrics; SSIM; MAE_metrics')
%disp([num2str(A1),'; ',num2str(B1),'; ',num2str(C1),'; ',num2str(D1),'; ',num2str(E1)])
fprintf('Average(mask 0-10)--PSNR_metrics:%.2f; SSIM:%.4f; MAE_metrics:%.2f\n',C1,D1,E1)
%disp('Average(mask 10%-20%): PSNR_test; MAE_test; PSNR_metrics; SSIM; MAE_metrics')
%disp([num2str(A2),'; ',num2str(B2),'; ',num2str(C2),'; ',num2str(D2),'; ',num2str(E2)])
fprintf('Average(mask 10-20)--PSNR_metrics:%.2f; SSIM:%.4f; MAE_metrics:%.2f\n',C2,D2,E2)
%disp('Average(mask 20%-30%): PSNR_test; MAE_test; PSNR_metrics; SSIM; MAE_metrics')
%disp([num2str(A3),'; ',num2str(B3),'; ',num2str(C3),'; ',num2str(D3),'; ',num2str(E3)])
fprintf('Average(mask 20-30)--PSNR_metrics:%.2f; SSIM:%.4f; MAE_metrics:%.2f\n',C3,D3,E3)
%disp('Average(mask 30%-40%): PSNR_test; MAE_test; PSNR_metrics; SSIM; MAE_metrics')
%disp([num2str(A4),'; ',num2str(B4),'; ',num2str(C4),'; ',num2str(D4),'; ',num2str(E4)])
fprintf('Average(mask 30-40)--PSNR_metrics:%.2f; SSIM:%.4f; MAE_metrics:%.2f\n',C4,D4,E4)
%disp('Average(mask 40%-50%): PSNR_test; MAE_test; PSNR_metrics; SSIM; MAE_metrics')
%disp([num2str(A5),'; ',num2str(B5),'; ',num2str(C5),'; ',num2str(D5),'; ',num2str(E5)])
fprintf('Average(mask 40-50)--PSNR_metrics:%.2f; SSIM:%.4f; MAE_metrics:%.2f\n',C5,D5,E5)
%disp('Average(mask 50%-60%): PSNR_test; MAE_test; PSNR_metrics; SSIM; MAE_metrics')
%disp([num2str(A6),'; ',num2str(B6),'; ',num2str(C6),'; ',num2str(D6),'; ',num2str(E6)])
fprintf('Average(mask 50-60)--PSNR_metrics:%.2f; SSIM:%.4f; MAE_metrics:%.2f\n',C6,D6,E6)
%disp('Average(mask 60%-70%): PSNR_test; MAE_test; PSNR_metrics; SSIM; MAE_metrics')
%disp([num2str(A7),'; ',num2str(B7),'; ',num2str(C7),'; ',num2str(D7),'; ',num2str(E7)])
fprintf('Average(mask 60-70)--PSNR_metrics:%.2f; SSIM:%.4f; MAE_metrics:%.2f\n',C7,D7,E7)

