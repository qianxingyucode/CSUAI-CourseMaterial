DATAREA SEGMENT
   mess1 db "Enter keyword:",'$'
   mess2 db "Enter sentence:",'$'
   mess3 db "Match at location:",'$'
   mess4 db "No match!",13,10,'$'
   mess5 db "match",13,10,'$'
   mess6 db "H of the sentence",13,10,'$'
   ;
   stok1 label byte
   max1 db 10
   act1 db ?
   stocks1 db 10 dup(?)
   ;
   stok2 label byte
   max2 db 50
   act2 db ?
   stocks2 db 50 dup(?)
DATAREA ENDS

PROGNAM segment
;

MAIN PROC FAR
     ASSUME CS:PROGNAM ,DS:DATAREA,ES:DATAREA
    start:
     PUSH DS
     SUB AX,AX
     SUB BX,BX
     PUSH AX
     mov AX,DATAREA
     MOV DS,AX
     MOV ES,AX
     ;此处开始进行字符串的对比
     ;先显示第一行 关键字
     lea dx,mess1
    mov ah,09;ds:dx显示字符串
     int 21h
     lea dx, stok1
     mov ah,0ah;ds:dx 键盘输入到缓冲区
     int 21h
     cmp act1,0
     je exit;如果未输入即关键字为空时直接跳出
    
    ;此时需要回车换行
step1:  
    call crlf
    lea dx,mess2
    mov ah,09
    int 21h
    lea dx,stok2
    mov ah,0ah
    int 21h
    cmp act2,0
    je nomatch
    mov al,act1
    cbw
    mov cx,ax;cx里装着act1的值
    push cx;cx是需要比较的keyword的长度 
    mov al,act2
    inc al
    sub al,act1;如果输入的字符句子小于关键字则肯定也不匹配
    js nomatch;结果为负则直接转移到不匹配
    mov di,0
    mov si,0
    lea bx,stocks2
   
    
 step2:
    mov ah,[bx+di]
    cmp ah,stocks1[si]
    jne step3
    inc si
    inc di
    dec cx
    cmp cx,0
    je match
    jmp step2;还未比较完就继续比较
 step3:
       inc bx;换下一组字符进行比较
       dec al
       cmp al,0
       je nomatch
       mov si,0
       mov di,0
       pop cx
       push cx
      ;这一步视为恢复cx的值
       jmp step2
  exit:
      call crlf
      ret
  nomatch:
      call crlf
      lea dx,mess4
      mov ah,09
      int 21h
      jmp step1;///////重新输入一个需要可能含关键字的字符串
  match:
      call crlf
      lea dx,mess3
       mov ah,09
       int 21h
       sub bx,offset stocks2
       inc bx
       call trans;此时需要进行二进制到十六进制的转换
       lea dx,mess6
       mov ah,09
       int 21h
       jmp step1
       
crlf proc near;回车换行
          mov dl,0dh
          mov ah,2
          int 21h
          mov dl,0ah
          mov ah,2
          int 21h
          ret
crlf endp          
          
 trans proc near;二进制转换为十六进制
   mov ch,4
   rotate:
   mov cl,4
   rol bx,cl
   mov al,bl
   and al,0fh
   add al,30h
   cmp al,3ah
   jl printit
   add al,07h
  printit:
       mov dl,al
       mov ah,2
       int 21h
       dec ch
       jnz rotate
       ret
       trans endp
       main endp
       prognam ends 
       end main


