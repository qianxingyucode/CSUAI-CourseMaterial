DATA segment
 result db 1d,24 dup (0)
 x db 1d,24 dup (0)
 
 mess1 db 0dh,0ah,'Please choose a data from 1 to 100:','$'
 mess2 db 0dh ,0ah,' The result is :','$'
 mess3 db 0dh ,0ah,' Press Q to exit.','$' 
 flag  dw ?
 mess4 db 0dh ,0ah,'Iuput error , please enter again.','$'
 mess5 db 0dh ,0ah,'exit.','$'
DATA ends 

stack segment ;�˴������ջ�δ���
 dW 128 dup (?)
stack ends
 
CODE segment 
 assume cs:CODE, ds:DATA, ss:stack, es : DATA
start :
 push ds 
 sub ax,ax
 sub bx,bx
 push ax	
 mov ax , DATA 
 mov ds , ax 
 mov es , ax 
 mov ax , stack
 mov ss , ax 

reinput :
 push ax 
 push bx
 push cx
 push dx
 push si
 push di
 lea dx,mess3 
 mov ah,09
 int 21h          ;��ʾ' Press Q to exit.'
 lea  dx,mess1
 mov ah,09
 int 21h          ;��ʾ'Please choose a data from 1 to 100:'
 mov bx,0
input:     ;����n,��ת��Ϊʮ������
   mov ah,01
   int 21h
   cmp al,'Q'
   jz end1      ;������Q,���˳�
   cmp al,0dh
   jz count      ;�ж��Ƿ��ǻس�(ʮ�������Ƿ�ת�����),������ʼ����fibonacci��
   cmp al,'0'
   jb  error
   sub al,30h  
   cbw
   xchg ax,bx   
   mov cx ,10d
   mul cx 
   xchg ax,bx 
   add bx,ax       ;��ʮ��������ת��Ϊʮ������
   jmp input 
error: ;�������  
   lea dx,mess4
   mov ah,09 
   int 21h
   jmp reinput ;��ת�ؿ�ʼ״̬
end1:    ; �˳�����
  lea dx,mess5
  mov ah,09
  int 21h       ;��ʾ' You have typed Q to exit.'
  mov ah ,4ch
  int 21h
  ret 
count : 
   mov cx,bx ;����������ŵ� cx ��
   cmp cx ,2
   jle print ;���С�ڵ���2,��ֱ��������
   sub cx ,2;������cx-2��Ϊѭ��,���������
next : 
   mov di,cx 
   mov cx,25   ;ѭ������
   mov si,0
add1 : 
   mov  dl,x[ si ]
   mov  dh,result[ si ]
   add  dl,dh ;�������洢��Ԫ�����н������
   mov  result[si],dl
   mov  x[si],dh ;���ϴε���ӽ������x[si]��
   cmp   dl,10d
   jae   great ;�������10D
   inc   si 
   jmp goon 
great : 
   sub   result[si],10d ;��β������result[si]
   inc   si
   add   x[si],1;��λ��1 
goon :
   loop add1
   mov cx,di 
   loop next 
print : 
    lea dx ,mess2  ;��ʾ' The result is :'
    mov ah ,09
    int 21h
    mov cx,25
    mov si,24
display1:
    cmp flag,0;��־λ�ж�����ĸ�λ�Ƿ�Ϊ0
    jnz if_nz
    cmp result [si],0
    jz  if_z
    add flag,1
if_nz: 
   mov  dl,result[si] ;��ʮ�������
   add dl ,30h
   mov ah ,02h
   int 21h
if_z: 
   dec si
   loop display1
   mov flag ,0  
   mov result[0],1d
   mov x[0],1d
   mov si,1
   mov cx,24      
initial:  ;��ʼ��
   mov result[si],0
   mov x[si],0
   add si,1
   loop initial
   mov si,0
   pop di
   pop si
   pop dx
   pop cx
   pop bx
   pop ax
   jmp reinput;��ת����ʼ״̬
   
   mov ah,4ch
   int 21h
  
 CODE ends
   end start

