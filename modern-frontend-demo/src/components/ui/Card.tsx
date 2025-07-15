import React, { forwardRef } from 'react'
import { motion } from 'framer-motion'
import { cn } from '@/utils/helpers'

export interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  hoverable?: boolean
}

const Card = forwardRef<HTMLDivElement, CardProps>(
  ({ className, hoverable = false, children, ...props }, ref) => {
    const Component = hoverable ? motion.div : 'div'
    
    return (
      <Component
        ref={ref}
        className={cn(
          "glass-card rounded-2xl p-6 shadow-lg",
          hoverable && "hover-lift cursor-pointer",
          className
        )}
        {...(hoverable && {
          whileHover: { scale: 1.02 },
          whileTap: { scale: 0.98 }
        })}
        {...props}
      >
        {children}
      </Component>
    )
  }
)

Card.displayName = "Card"

const CardHeader = forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div
      ref={ref}
      className={cn("flex flex-col space-y-1.5 pb-6", className)}
      {...props}
    />
  )
)

CardHeader.displayName = "CardHeader"

const CardTitle = forwardRef<HTMLParagraphElement, React.HTMLAttributes<HTMLHeadingElement>>(
  ({ className, ...props }, ref) => (
    <h3
      ref={ref}
      className={cn(
        "text-2xl font-bold leading-none tracking-tight gradient-text",
        className
      )}
      {...props}
    />
  )
)

CardTitle.displayName = "CardTitle"

const CardContent = forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div ref={ref} className={cn("space-y-4", className)} {...props} />
  )
)

CardContent.displayName = "CardContent"

export { Card, CardHeader, CardTitle, CardContent }