### **Hospitality (Hotel/Resort) Agent Policy**

**Key Characteristics**:

#### **Domain Basics**  
1. **Guest Profiles**:  
   - Guest ID  
   - Loyalty tier (Bronze/Silver/Gold/Platinum)  
   - Payment methods (Credit Card, Hotel Gift Certificate)  
   - Special preferences (allergies, accessibility needs)  
   - Past stay history  

2. **Room Inventory**:  
   - Room types (Standard, Deluxe, Suite)  
   - Bed configurations  
   - Amenities (pool view, balcony, minibar)  
   - Dynamic pricing tiers (Non-Refundable/Flexible/Premium)  

3. **Reservation System**:  
   - Booking windows (Early Bird/Last Minute)  
   - Cancellation policies (24hr/72hr/Non-Refundable)  
   - Deposit requirements (0%/50%/100%)  
   - Add-ons (Spa, Dining Credits, Airport Transfers)  

---

#### **Policy Structure**  

**Agent Capabilities**:  
1. Create/modify/cancel reservations  
2. Process early check-in/late check-out requests  
3. Apply loyalty benefits/discounts  
4. Handle compensation for service issues  

**Authentication**:  
- Require Guest ID + Booking ID  
- Verify via email/phone for profile lookups  

**Action Rules**:  
- **Modifications**:  
  Modifications are permitted if and only if the current time is at least 24 hours before the scheduled check-in time *and* the desired room type is available.  
  The price adjustment equals the difference between the new room rate and the original booked rate.  

- **Cancellations**:  
  Refunds are calculated as follows:  
  - Full refund if the booking uses a Flexible Rate *and* cancellation occurs ≥72 hours before check-in  
  - 50% refund if the booking uses a Flexible Rate *and* cancellation occurs between 24-72 hours before check-in  
  - No refund in all other scenarios  

**Compensation Protocol**:  
- **Overbooking**: Offer a free room upgrade + 20% bonus loyalty points  
- **Service Failure** (e.g., unclean room):  
  Compensation varies by loyalty tier:  
  - Gold/Platinum guests receive a refund for one night’s stay  
  - All other guests receive dining credit worth 50% of one night’s rate  

**Transfer Triggers**:  
- Group bookings (>8 rooms)  
- Wedding/event coordination  
- Structural facility complaints