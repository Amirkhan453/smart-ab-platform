#include <stdint.h>
#include <stdio.h>
#include <string.h>
static uint32_t rotl(uint32_t x,int r){return (x<<r)|(x>>(32-r));}
uint32_t mm(const uint8_t*d,int len,uint32_t seed){
 int nb=len/4; uint32_t h=seed,c1=0xcc9e2d51,c2=0x1b873593;
 for(int i=0;i<nb;i++){uint32_t k; memcpy(&k,d+i*4,4); k*=c1;k=rotl(k,15);k*=c2;h^=k;h=rotl(h,13);h=h*5+0xe6546b64;}
 const uint8_t*t=d+nb*4; uint32_t k=0;
 switch(len&3){case 3:k^=t[2]<<16; case 2:k^=t[1]<<8; case 1:k^=t[0];k*=c1;k=rotl(k,15);k*=c2;h^=k;}
 h^=len; h^=h>>16;h*=0x85ebca6b;h^=h>>13;h*=0xc2b2ae35;h^=h>>16; return h;}
int main(){char line[4096]; while(fgets(line,sizeof line,stdin)){int n=strlen(line); if(n&&line[n-1]=='\n')line[--n]=0; printf("%u\n",mm((uint8_t*)line,n,0));}
 /* smhasher verification */
 uint8_t key[256],hashes[1024]; for(int i=0;i<256;i++){key[i]=i; uint32_t h=mm(key,i,256-i); memcpy(hashes+i*4,&h,4);} fprintf(stderr,"verif=%08X\n",mm(hashes,1024,0)); return 0;}
