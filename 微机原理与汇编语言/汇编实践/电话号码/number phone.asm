;�ȴ湻�绰�����ٿ�ʼ��ʾ����
dataarea  segment 
mess1    db    'Please input name:','$' 
mess2    db    'Please input telephone number:','$' 
mess3    db    'Do you want to search a telephone number?(y/n)','$' 
mess4    db    0dh,0ah,'what is the name?','$' 
mess5    db    'Not find',0dh,0ah,'$' 
mess6    db    'the number you want to store:50',13,10,'$' 
crlf     db     0dh,0ah,'$' 


ArrayName  label  byte                       ;������������
max1      db     21
act1      db     ?
Name_char    db     21 dup(?)
                                             
ArrayTel  label  word                        ;���յ绰��������
max2      db     9
act2      db     ?
Telchar   db    9 dup(?)
 
Tel_Tab   db     50 dup(28 dup(?))      ;�ַ����������ı��ڸ��Ӷ�

count    db 0
name_count dw    0
endAddress   dw  ?
swapped   dw     ?                      ;����λswapped���ж��Ƿ��������
TotalNumber dw   50
tempSave    db   28 dup(?),0dh,0ah,'$'  ;�ݴ�Tel_Tab��һ��
FindAdd dw    ?
flag      db     ?
show      db     'name                phone',0dh,0ah,'$' ;��ʾ
dataarea  ends
;
codesg  segment
   assume ds:dataarea,cs:codesg,es:dataarea
main    proc    far
       mov ax,dataarea 
       mov ds,ax 
       mov es,ax 
       lea di,Tel_Tab     ;�����ַ����͵绰����������� 
       lea dx,mess6        ;��ʾ��Ϣ�������������ĵ绰�����ܸ��� 50
       mov ah,09   
       int 21h             ;OUTPUT DX
 Mstep1:                              
       lea dx,mess1                 ;��ʾ����������
       mov ah,09 
       int 21h 
       call input_name              ;�������� 
       inc  name_count              ;������++
       call stor_name               ;�����ִ���洢�ռ�
       lea dx,mess2                 ;��ʾ������绰
       mov ah,09 
       int 21h 
       call input_Tel                 ;�������
       call stor_phone              ;�洢���� 
       cmp  name_count,0   
       je  exit 
       mov bx,TotalNumber           ;������������BX
       cmp  name_count,bx           ;����50����������绰�����ֶ�ѭ��
       jnz  Mstep1 
       call name_sort 
 Mstep2:
       lea dx,mess3                  ;��ʾ�Ƿ���Ҫ���ҵ绰����
       mov ah,09 
       int 21h
       mov ah,08                     ;��������x��y����ѡ���޻��ԣ�AL=�����ַ�
       int 21h
       cmp al,'y'
       jz  Mstep3
       cmp al,'n'
       jz  exit
       jmp Mstep2                          
 Mstep3:
       mov ah,09
       lea dx,mess4                  ;��ʾѯ������
       int 21h                       ;����ǰ����Ҫ����������
       call input_name      
 Mstep4:
       call FindName              ;��ʼ����
       jmp Mstep2                    
 exit:
       mov ax,4c00h            ;������ֹ
       int 21h
 main endp
;--------------------------------------------------------------------
input_name  proc  near
     mov ah,0ah
     lea dx,ArrayName       ;�����������ַ��DX
     int 21h
     mov ah,09              ;��������
     lea dx,crlf            ;�س�����
     int 21h
     sub bh,bh              ;bh����,BL�泤��
     mov bl,act1            ;����16����λ�ܴ���ACT1
     mov cx,21
     sub cx,bx              ;CX=CX-BXΪ��ȱ����,CXΪ������
b10:
     mov Name_char[bx],' '   ;��ȫ�ո�
     inc bx
     loop b10
    ret
input_name endp
;--------------------------------------------------------------------
stor_name     proc   near
      lea  si,Name_char
      mov  cx,20
      rep  movsb ;movsbָ�����ڰ��ֽڴ�ds:si �ᵽes:di���ӶΣ�rep��repeat����˼��rep movsb ���Ƕ�ΰ��ˡ�
                 ;����ǰ�Ȱ��ַ����ĳ��ȴ���cx�Ĵ����У�Ȼ���ظ��Ĵ�������cx�Ĵ����������ݵ�ֵ��
      ret
stor_name  endp
;--------------------------------------------------------------------
input_Tel   proc   near
     mov ah,0ah 
     lea dx,ArrayTel 
     int 21h 
     mov ah,09         ;��绰����
     lea dx,crlf       ;���лس�
     int 21h 
     sub bh,bh         ;ԭ��ͬ�洢������
     mov bl,act2 
     mov cx,9  
     sub cx,bx 
c10:
     mov Telchar[bx],' ' ;����ո�
     inc bx              ;
     loop c10
     ret 
input_Tel endp
;--------------------------------------------------------------------
stor_phone  proc near
     lea  si,Telchar
     mov  cx,8
     rep  movsb   ;movs ������ָ�� cmps ���Ƚϲ���
     ret
stor_phone endp
;--------------------------------------------------------------------
name_sort  proc near        ;ð�ݷ�������+�绰���룬������
     sub  di,28             ;��ʱDIָ�����һ���ַ������׵�ַ
     mov  endAddress,di        ;DI-28->������ַ
 c1:
     mov  swapped,0            ;swap��0��ʼ
     lea  si,Tel_Tab 
 c2: 
     mov  cx,20 
     mov  di,si 
     add  di,28   ;����DI=���ַ+28Ҳ������һ���������  ��SI��DI��Ϊ������ַָ�룬ָ��ES��
     mov  ax,di 
     mov  bx,si   ;cmpsb si-di  movsb di<-si
     repz cmpsb   ;repz ��Ϊ0ʱ�ظ���������
     jbe  if_Loop ;CF��ZF=1��С�ڵ�����ת��
                ;repz��һ��������ǰ׺�����ظ�������ָ�ÿ�ظ�һ��ECX��ֵ�ͼ�һ
                ;һֱ��CXΪ0��ZFΪ0ʱֹͣ��
                ;�����򽻻�λ��
     mov si,bx    
     lea di,tempSave ;DIָ��TempSave
     mov cx,28
     rep movsb       ;ת�Ƶ�tempSave��
     mov cx,28                   ;��ds:si �ᵽes:di���Ӷ�
     mov di,bx       ;DIָ��BX
     rep movsb       ;�ᵽBX
     mov cx,28
     lea si,tempSave 
     rep movsb       ;tempSave���ȥ
     mov swapped,1
 if_Loop:
     mov  si,ax
     cmp  si,endAddress ;������û�б��꣨һ�ֵĴ�ͷ��β��
     jb   c2            ;С����CF=1����ת�ƣ�
     cmp  swapped,0      ;CMP - ���������ֵ��ȣ���Z��־������Ϊ��1����������û�б����ã�0��
     jnz  c1             ;�����Ϊ0��ת��
     ret 
name_sort endp
;--------------------------------------------------------------------
FindName proc near
      lea  bx,Tel_Tab
	mov  flag,0      
   dStep1: 
      mov  cx,20 
	lea si,Name_char 
      mov  di,bx            ;���ո�Ϊ�˷��������ͱȽ�
      repz cmpsb            ;BX���ű�
      jz  Found
      add bx,28             ;ָ����һ����������û��
      cmp  bx,endAddress    ;endAress�Ǳ�βָ�� 
      jbe  dStep1           ;û��β��δ������������
	sub flag,0            ;Ҫ��û���ҵ��Ļ�
      jz NotFound
      jmp  dexit            ;�������˳�
  NotFound:  
      lea dx,mess5
      mov ah,09
      int 21h 
  Found:
      mov FindAdd,bx       ;  BX��Ӧ�ҵ��ĵ�ַ��FindAdd
	inc flag
	call printf
	add bx,28            ;��һ��
      cmp  bx,endAddress   ;�Ƿ������
      jbe  dStep1          ;δ������������
      jmp  dexit           ;�������˳�
      jnz  dStep1
 dexit:
        ret
FindName endp
;--------------------------------------------------------------------
printf proc  near
       sub flag,0     ;Ҫ��û���ҵ��Ļ�������Notfind
       jz  Nofind

 pStep:
       mov ah,09
       lea dx,show
       int 21h
       mov cx,28
       mov si,FindAdd
       lea di,tempSave
       rep movsb
       lea dx,tempSave
       mov ah,09
       int 21h
       jmp FinalExit

Nofind:    
       lea dx,mess5
       mov ah,09
       int 21h 
FinalExit:  
       ret
printf endp

codesg ends

end main       
